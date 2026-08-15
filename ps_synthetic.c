//kv260 project: parallel FOL inference on PL
//potential use: parallel AG generation on PL

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>
#include <time.h>


// FOL control parameters start
int NUM_CONSTANTS;          // 75  //25  //50
#define NUM_UNARY               50  //200 //1000
#define NUM_BINARY              10  //200 //500
#define UNARY_DENSITY           0.1  //0.1 //0.8
#define FULL_COVERAGE_FIRST_N   5  //5 //10
#define NUM_HUB_CONSTANTS       2  //2 //5

#define RULES_LAYER1  50 //50 //1000
#define RULES_LAYER2  400 //400 //1000
#define RULES_LAYER3   400 //400//1000
#define RULES_LAYER4   200 //200 //1000
#define RULES_LAYER5   200 //200 //500
#define RULES_LAYER6   100 //100 //500

// predicate arguments encoded >= VB are variables, <VB are constants
#define VB 32768

// hash table capacity (# of facts)
#define HS_CAP 65536 //49152

#define MAX_ARITY       4           // pred_id 1 + up to 3 args = 4
#define MAX_RULES       (RULES_LAYER1+RULES_LAYER2+RULES_LAYER3+RULES_LAYER4+RULES_LAYER5+RULES_LAYER6)
#define RULE_LIMIT      2048
#define MAX_PREMISES    4           // max number of predicates in premise of a rule
#define MAX_NEWCNT      1024        /* iteration log depth */

/* Max permutations: argc ≤ 3, NUM_CONSTANTS=40 → 40^3 = 64 000 */
#define MAX_PERMS       150000

/* ============================================================
   Simple LCG random number generator
   (reproduces Python random.seed / random.sample behaviour for
    uniform sampling without replacement)
   ============================================================ */
static unsigned long long lcg_state;

static void lcg_seed(unsigned long long s) { lcg_state = s; }

/* Returns a uniform random integer in [0, n) */
static int lcg_randint(int n) {
    lcg_state = lcg_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((lcg_state >> 33) % (unsigned long long)n);
}

/* Fisher-Yates sample of `k` distinct values from arr[0..n-1].
   Result stored in out[0..k-1]. */
static void lcg_sample(int *arr, int n, int k, int *out) {
    /* copy arr into a temp buffer, then partial Fisher-Yates */
    int *tmp = (int *)malloc(n * sizeof(int));
    memcpy(tmp, arr, n * sizeof(int));
    for (int i = 0; i < k; i++) {
        int j = i + lcg_randint(n - i);
        int t = tmp[i]; tmp[i] = tmp[j]; tmp[j] = t;
        out[i] = tmp[i];
    }
    free(tmp);
}

// FOL control parameters done

/* ============================================================
   Tuple  (pred_id, arg0, arg1, …)
   ============================================================ */
typedef struct {
    uint16_t v[MAX_ARITY];
} Tuple;

static int tuple_eq(const Tuple *a, const char len_a, const Tuple *b, const char len_b) {
    if (len_a != len_b) return 0;
    for (int i = 0; i < len_a; i++)
        if (a->v[i] != b->v[i]) return 0;
    return 1;
}

static Tuple make_tuple1(uint16_t p, uint16_t a) {
    Tuple t; t.v[0]=p; t.v[1]=a; t.v[2]=0; t.v[3]=0; return t;
}
static Tuple make_tuple2(uint16_t p, uint16_t a, uint16_t b) {
    Tuple t; t.v[0]=p; t.v[1]=a; t.v[2]=b; t.v[3]=0; return t;
}
static Tuple make_tuple3(uint16_t p, uint16_t a, uint16_t b, uint16_t c) {
    Tuple t; t.v[0]=p; t.v[1]=a; t.v[2]=b; t.v[3]=c; return t;
}

/* ============================================================
   Hash set  (open-addressing, linear probing)
   Heap-allocated so large HS_CAP doesn't blow the stack.
   ============================================================ */
typedef struct {
    Tuple  *slots;
    char   *used; //used = 0 means unused, used!=0 indicates currently used and the length of the Tuple (predicate)
    int    *count;
} HashSet;

static unsigned int tuple_hash(const Tuple *t, const char len) {
    unsigned int h = 2166136261u;
    for (int i = 0; i < len; i++) {
        h ^= (unsigned int)t->v[i];
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
    }
    return h;
}

//used to initialize the hash table of current predicates (facts)
static void hs_init(HashSet *hs, void *addr, void *count_addr) {
    hs->slots = (Tuple *)addr;
    hs->used  = (char  *)addr + HS_CAP*sizeof(Tuple);
    hs->count = (int *)(count_addr);
    memset(addr, 0, HS_CAP * sizeof(Tuple) + HS_CAP * sizeof(char));
    hs->count[0] = 0;
}

//used to initialize the hash table of new predicates (facts)
//sitting on top of the current predicate hash table in URAM space
static void hs_init_new(HashSet *hs, void *addr, void *count_addr) {
    hs->slots = (Tuple *)((char*)addr + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char));
    hs->used  = (char  *)addr + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char) + HS_CAP*sizeof(Tuple);
    hs->count = (int *)(count_addr);
    memset((void *)(hs->slots), 0, HS_CAP * sizeof(Tuple) + HS_CAP * sizeof(char));
    hs->count[0] = 0;
}

static void hs_free(HashSet *hs) {
    //free(hs->slots); free(hs->used);
    hs->slots = NULL; hs->used = NULL; //hs->count = NULL;
}

/* Returns 1 if newly inserted, 0 if already present */
static int hs_insert(HashSet *hs, const Tuple *t, const char len) {
    unsigned int h = tuple_hash(t, len) % HS_CAP;
    while (hs->used[h]) {
        if (tuple_eq(&hs->slots[h], hs->used[h], t, len)) return 0;
        h = (h + 1) % HS_CAP;
    }
    hs->slots[h] = *t;
    hs->used[h]  = len; //used unit indicates both used and length of stored predicate
    hs->count[0]++;
    return 1;
}

static int hs_contains(const HashSet *hs, const Tuple *t, const char len) {
    unsigned int h = tuple_hash(t, len) % HS_CAP;
    while (hs->used[h]) {
        if (tuple_eq(&hs->slots[h], hs->used[h], t, len)) return 1;
        h = (h + 1) % HS_CAP;
    }
    return 0;
}

static int hs_merge(HashSet *dst, const HashSet *src) {
    int x = 0;
    for (int i = 0; i < HS_CAP; i++)
        if (src->used[i]){ 
		hs_insert(dst, &src->slots[i], src->used[i]);
		x++;
	}
    return x;
}

/* ============================================================
   Rule representation
   ============================================================ */
typedef struct {
    Tuple clauses[MAX_PREMISES];
    uint64_t   n; //used to store the number of clauses in this clause set (byte0) and each clause's arity (byte7-4)
} ClauseSet;

typedef struct {
    ClauseSet premises;
    ClauseSet conclusions;
} Rule;

/*constants in FOL*/
static int *consts;
static int *nc;

static void const_init(void *addr1){
    consts = (int *)malloc(NUM_CONSTANTS*sizeof(int));
    nc = (int *)addr1;
}

/* Add a rule (premises + conclusion) to the rule array */
static int *rule_count;
static Rule *rules;
static uint8_t *rule_nargs;

static void rules_init(void *addr, void *addr1){
    rules = (Rule *)((char *)addr + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2); //positioned after two hash tables
    rule_nargs = (uint8_t *)((char *)addr + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule));
    rule_count = (int *)addr1;
    memset((void *)((char *)addr + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2), 0, RULE_LIMIT*sizeof(Rule)+32768);
    rule_count[0] = 0;
}

static uint16_t *active;
static int *n_active;

static void active_init(void *addr, void *addr1){
    active = (uint16_t *)((char *)addr + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule)+ 32768);
    memset((void *)active, 0, 32768);
    n_active = (int *)addr1;
    n_active[0] = 0;
}

static void add_rule(uint8_t argc,
                     int np, Tuple *prems, uint8_t *cnt_p, 
                     int nc2, Tuple *concs, uint8_t *cnt_c) {
    rule_nargs[*rule_count] =  argc;
    Rule *r = &rules[(*rule_count)++];
    
    for (int i = 0; i < np; i++) {
        r->premises.n = r->premises.n + cnt_p[i];
        r->premises.n = (r->premises.n)<<8;
    }
    r->premises.n = r->premises.n + (uint64_t)np;
    for (int i = 0; i < np; i++) r->premises.clauses[i]  = prems[i];
    
    for (int i = 0; i < nc2; i++) {
        r->conclusions.n = r->conclusions.n + cnt_c[i];
        r->conclusions.n = (r->conclusions.n)<<8;
    }
    r->conclusions.n = r->conclusions.n + (uint64_t)nc2;
    for (int i = 0; i < nc2; i++) r->conclusions.clauses[i] = concs[i];
}

typedef struct {
    int (*data)[MAX_ARITY-1];   /* heap: [MAX_PERMS][MAX_ARITY-1] */
    int count;
} PermSet;

static void _perm_rec(int *cur, int depth, int sz,
                       int *consts, int nc,
                       PermSet *out) {
    if (depth == sz) {
        for (int i = 0; i < sz; i++)
            out->data[out->count][i] = cur[i];
        out->count++;
        return;
    }
    for (int i = 0; i < nc; i++) {
        cur[depth] = consts[i];
        _perm_rec(cur, depth+1, sz, consts, nc, out);
    }
}

static void make_perm(int *consts, int nc, int sz, PermSet *out) {
    out->count = 0;
    int cur[MAX_ARITY-1];
    _perm_rec(cur, 0, sz, consts, nc, out);
}

static int merge_and_update(HashSet *facts, HashSet *new_facts, Rule *rules, int nrules)
{
        int x = hs_merge(facts, new_facts);
        if(x==0) return 0;
	

        /* recompute active rules */
        char in_next[MAX_RULES];
        memset(in_next, 0, sizeof(in_next));
        uint16_t next_active[MAX_RULES];
        int n_next = 0;

        for (int ni = 0; ni < HS_CAP; ni++) {
            if (!new_facts->used[ni]) continue;
            uint16_t pred = new_facts->slots[ni].v[0];
            for (int r = 0; r < nrules; r++) {
                if (in_next[r]) continue;
                int np3 = rules[r].premises.n&0x000000FF;
                for (int pi2 = 0; pi2 < np3; pi2++) {
                    if (rules[r].premises.clauses[pi2].v[0] == pred) {
                        in_next[r] = 1;
                        next_active[n_next++] = r;
                        break;
                    }
                }
            }
        }
        memcpy(active, next_active, n_next * sizeof(uint16_t));
        *n_active = n_next;

	return x;
}


//debugging ports mapped to GPIO
#define FSM_base1 0xB0010000
#define FSM_base2 0xB0020000
#define FSM_base3 0xA0000000
//dma_go ports: used by PS to trigger each cycle that PL writes data to DMA
#define FSM_base4 0xA0010000
//#define FSM_base5 0xA0020000
#define FSM_base6 0xA0030000
#define FSM_base7 0xA0040000
#define MAP_SIZE 128

/* ------------------------------------------------------------------ */
/* Board / design constants — adjust to match your Vivado address map  */
/* ------------------------------------------------------------------ */
//TX channel: PS write, PL read
//  Rules and other inputs
//  Current fact hash table
//RX channel: PL write, PS read
//  new fact list
#define DMA_BASE        0xB0000000UL   /* AXI DMA base address for buffer 1*/
#define DMA_MAP_SIZE    0x10000UL      /* 64 KB — covers all DMA regs   */

//#define TRANSFER_WORDS  524288 //2^19 words
//#define TRANSFER_BYTES  (TRANSFER_WORDS * sizeof(uint32_t)) //2^21 bytes = 2 MB

#define TRANSFER_WORDS  176128 //44 blocks //262144 //2^18 doublewords
#define TRANSFER_BYTES  (TRANSFER_WORDS * sizeof(uint32_t) * 2) //2^21 bytes = 2 MB



/* ------------------------------------------------------------------ */
/* AXI DMA register offsets (PG021)                                    */
/* ------------------------------------------------------------------ */
/* MM2S channel (memory → stream, i.e. transmit) */
#define MM2S_DMACR      0x00   /* Control                              */
#define MM2S_DMASR      0x04   /* Status                               */
#define MM2S_SA         0x18   /* Source Address (low 32 bits)         */
#define MM2S_SA_MSB     0x1C   /* Source Address (high 32 bits)        */
#define MM2S_LENGTH     0x28   /* Transfer length (bytes)              */

/* S2MM channel (stream → memory, i.e. receive) */
#define S2MM_DMACR      0x30   /* Control                              */
#define S2MM_DMASR      0x34   /* Status                               */
#define S2MM_DA         0x48   /* Destination Address (low 32 bits)    */
#define S2MM_DA_MSB     0x4C   /* Destination Address (high 32 bits)   */
#define S2MM_LENGTH     0x58   /* Transfer length (bytes)              */

/* DMACR bits */
#define DMACR_RS        (1u << 0)   /* Run/Stop                        */
#define DMACR_RESET     (1u << 2)   /* Soft reset (self-clearing)      */
#define DMACR_IOC_IrqEn (1u << 12) /* IOC interrupt enable            */
#define DMACR_ERR_IrqEn (1u << 14) /* Err interrupt enable            */

/* DMASR bits */
#define DMASR_HALTED    (1u << 0)   /* DMA halted                      */
#define DMASR_IDLE      (1u << 1)   /* DMA idle (transfer complete)    */
#define DMASR_SGINCLD   (1u << 3)   /* Scatter-gather included         */
#define DMASR_DMAINTERR (1u << 4)   /* Internal DMA error              */
#define DMASR_DMASLVERR (1u << 5)   /* Slave error                     */
#define DMASR_DMADECERR (1u << 6)   /* Decode error                    */
#define DMASR_IOC_Irq   (1u << 12)  /* IOC interrupt asserted          */
#define DMASR_ERR_Irq   (1u << 14)  /* Error interrupt asserted        */

#define DMASR_ANY_ERR   (DMASR_DMAINTERR | DMASR_DMASLVERR | DMASR_DMADECERR)

/* Convenience register accessors */
#define REG_RD(base, off)        (*((volatile uint32_t *)((uint8_t *)(base) + (off))))
#define REG_WR(base, off, val)   (*((volatile uint32_t *)((uint8_t *)(base) + (off))) = (val))

/* ------------------------------------------------------------------ */
/* Global state                                                        */
/* ------------------------------------------------------------------ */
static volatile void  *dma_regs  = MAP_FAILED;
static void           *rx_virt   = MAP_FAILED;   /* S2MM destination buffer */
static void           *tx_virt   = MAP_FAILED;   /* MM2S source buffer      */

static uint64_t rx_bus_addr = 0;
static uint64_t tx_bus_addr = 0;

static int  devmem_fd  = -1;
static int  heap_fd    = -1;
static int  buf_fd     = -1;   /* S2MM buffer fd */
static int  tx_heap_fd = -1;
static int  tx_buf_fd  = -1;   /* MM2S buffer fd */


int fd1;
volatile uint32_t *reg1, *reg2, *reg3, *reg4, *reg6, *reg7;
volatile uint32_t reg5[3] = {0, 0, 0};


/* ------------------------------------------------------------------ */
/* Helper: translate a locked virtual page to its physical address     */
/* via /proc/self/pagemap.                                             */
/*                                                                     */
/* Requires: CAP_SYS_ADMIN (root) and the buffer must be mlock()'d    */
/* first so the kernel has wired it to a physical page.               */
/*                                                                     */
/* On a KV260 without an SMMU (or with SMMU in bypass for PL),        */
/* physical address == bus address seen by the DMA engine.            */
/* ------------------------------------------------------------------ */
static int virt_to_phys(void *vaddr, uint64_t *phys_out)
{
    int fd = open("/proc/self/pagemap", O_RDONLY); //a read-only view of the pagetable for this program
    if (fd < 0) {
        perror("open /proc/self/pagemap");
        return -1;
    }

    long page_size = sysconf(_SC_PAGESIZE);
    uintptr_t vpage = (uintptr_t)vaddr / page_size; //vpn
    off_t offset    = (off_t)(vpage * 8); //each pagetable entry is 64-bit/8-byte

    if (lseek(fd, offset, SEEK_SET) < 0) {//repositioning to the pagetable entry for the vpn of vaddr
        perror("lseek pagemap");
        close(fd);
        return -1;
    }

    uint64_t entry = 0;
    if (read(fd, &entry, sizeof(entry)) != sizeof(entry)) {
        perror("read pagemap");
        close(fd);
        return -1;
    }
    close(fd);

    /* Bit 63: page present; bits 54:0: PFN */
    if (!(entry & (1ULL << 63))) {//is this page currently in physical memmory? possible not in due to virtual memory management
        fprintf(stderr, "Page not present in RAM — was mlock() called?\n");
        return -1;
    }

    uint64_t pfn = entry & ((1ULL << 55) - 1); //extract the lower 55 bits for the physical page(frame) number
    *phys_out = (pfn * page_size) + ((uintptr_t)vaddr % page_size); //assemble the full physical address
    return 0;
}

/* ------------------------------------------------------------------ */
/* 1. Map AXI DMA registers via /dev/mem                              */
/* ------------------------------------------------------------------ */
static int init_dma_regs(void)
{
    devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd < 0) {
        perror("open /dev/mem");
        fprintf(stderr,
            "Hint: needs root and CONFIG_STRICT_DEVMEM=n (or UIO driver).\n");
        return -1;
    }

    dma_regs = mmap(NULL, DMA_MAP_SIZE,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    devmem_fd,
                    (off_t)DMA_BASE);

    if (dma_regs == MAP_FAILED) {
        perror("mmap DMA registers");
        return -1;
    }

    printf("[init] DMA registers mapped: phys=0x%08lX → virt=%p\n",
           DMA_BASE, (void *)dma_regs);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 2. Allocate a physically-contiguous DMA buffer via CMA heap         */
/* ------------------------------------------------------------------ */
static int init_rx_buffer(void)
{
    /*
     * Prefer the "reserved" CMA heap which is guaranteed contiguous.
     * Fall back to "system" if the BSP doesn't expose a reserved heap.
     */
    heap_fd = open("/dev/dma_heap/reserved", O_RDWR);
    if (heap_fd < 0) {
        fprintf(stderr,
            "[init] /dev/dma_heap/reserved not found, trying system heap\n");
        heap_fd = open("/dev/dma_heap/system", O_RDWR);
    }
    if (heap_fd < 0) {
        perror("open dma_heap");
        return -1;
    }

    /* Use the struct directly from <linux/dma-heap.h> — do NOT redefine it */
    struct dma_heap_allocation_data alloc; //a Linux struct type for DMA purpose
    memset(&alloc, 0, sizeof(alloc));
    alloc.len      = TRANSFER_BYTES; //this is where the size of the buffer for DMA is set
    alloc.fd_flags = O_RDWR | O_CLOEXEC; //allow the buffer to be read and written
    /* alloc.heap_flags and alloc.fd are zero-initialised (correct) */

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {//the ioctl is supposed to grant the request
        perror("DMA_HEAP_IOCTL_ALLOC");
        return -1;
    }
    buf_fd = (int)alloc.fd; //convert __u32 typed alloc.fd to int type

    /* Map the buffer into our virtual address space */
    rx_virt = mmap(NULL, TRANSFER_BYTES, //now we can use rx_virt to refer to this DMA buffer
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   buf_fd, 0);
    if (rx_virt == MAP_FAILED) {
        perror("mmap CMA buffer");
        return -1;
    }

    /*
     * mlock() the buffer so the kernel wires it to physical RAM before
     * we query the PFN via pagemap.
     */
    if (mlock(rx_virt, TRANSFER_BYTES) < 0) { //wire the buffer pages down to physical memory to prohibit swap
        perror("mlock CMA buffer");
        return -1;
    }

    /* Touch every page to force population (mlock alone may not fault them in) */
    memset(rx_virt, 0, TRANSFER_BYTES); //zero the entire buffer

    printf("[init] RX buffer: virt=%p, size=%u bytes\n",
           rx_virt, (unsigned)TRANSFER_BYTES);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 2b. Allocate a physically-contiguous TX buffer for MM2S            */
/* ------------------------------------------------------------------ */
static int init_tx_buffer(void)
{
    tx_heap_fd = open("/dev/dma_heap/reserved", O_RDWR);
    if (tx_heap_fd < 0) {
        fprintf(stderr,
            "[init] /dev/dma_heap/reserved not found, trying system heap\n");
        tx_heap_fd = open("/dev/dma_heap/system", O_RDWR);
    }
    if (tx_heap_fd < 0) {
        perror("open dma_heap (TX)");
        return -1;
    }

    struct dma_heap_allocation_data alloc;
    memset(&alloc, 0, sizeof(alloc));
    alloc.len      = TRANSFER_BYTES;
    alloc.fd_flags = O_RDWR | O_CLOEXEC;

    if (ioctl(tx_heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc) < 0) {
        perror("DMA_HEAP_IOCTL_ALLOC (TX)");
        return -1;
    }
    tx_buf_fd = (int)alloc.fd;

    tx_virt = mmap(NULL, TRANSFER_BYTES,
                   PROT_READ | PROT_WRITE,
                   MAP_SHARED,
                   tx_buf_fd, 0);
    if (tx_virt == MAP_FAILED) {
        perror("mmap CMA buffer (TX)");
        return -1;
    }

    if (mlock(tx_virt, TRANSFER_BYTES) < 0) {
        perror("mlock CMA buffer (TX)");
        return -1;
    }

    /* Pre-fill TX buffer with a known pattern the PL can receive */
    //uint32_t *tx = (uint32_t *)tx_virt;
    //for (int i = 0; i < TRANSFER_WORDS; i++)
    //    tx[i] = (uint32_t)i;   /* 0, 1, 2, … — change as needed */
    
    /* Touch every page to force population (mlock alone may not fault them in) */
    memset(tx_virt, 0, TRANSFER_BYTES); //zero the entire buffer    

    printf("[init] TX buffer: virt=%p, size=%u bytes\n",
           tx_virt, (unsigned)TRANSFER_BYTES);
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3. Software-reset S2MM channel and poll until self-clear           */
/* ------------------------------------------------------------------ */
static int reset_s2mm(void)
{
    REG_WR(dma_regs, S2MM_DMACR, DMACR_RESET);

    
    int timeout = 10000;
    while (REG_RD(dma_regs, S2MM_DMACR) & DMACR_RESET) {
        if (--timeout <= 0) {
            fprintf(stderr, "S2MM reset timed out!\n");
            return -1;
        }
        usleep(1);
    }
    printf("[dma] S2MM reset complete\n");
    return 0;
}

/* ------------------------------------------------------------------ */
/* 3b. Software-reset MM2S channel and poll until self-clear          */
/* ------------------------------------------------------------------ */
static int reset_mm2s(void)
{
    REG_WR(dma_regs, MM2S_DMACR, DMACR_RESET);

    int timeout = 10000;
    while (REG_RD(dma_regs, MM2S_DMACR) & DMACR_RESET) {
        if (--timeout <= 0) {
            fprintf(stderr, "MM2S reset timed out!\n");
            return -1;
        }
        usleep(1);
    }
    printf("[dma] MM2S reset complete\n");
    return 0;
}


//s2mm arming
static int s2mm(uint64_t bus_addr, uint32_t nbytes){
    uint32_t addr_lo = (uint32_t)(bus_addr & 0xFFFFFFFFULL);
    uint32_t addr_hi = (uint32_t)(bus_addr >> 32);

    REG_WR(dma_regs, S2MM_DA,     addr_lo);
    REG_WR(dma_regs, S2MM_DA_MSB, addr_hi);

    REG_WR(dma_regs, S2MM_DMACR,
           DMACR_RS | DMACR_IOC_IrqEn | DMACR_ERR_IrqEn);
    
    REG_WR(dma_regs, S2MM_LENGTH, nbytes);
    reg4[0] = 1; //go-ahead signal via GPIO for DMA_tester to start a full (Device) write cycle of 1024 words
    __sync_synchronize();
    reg4[0] = 0; //turn off right away to avoid later triggering the next cycle 
    __sync_synchronize();  
    return 0;  
}

//mm2s arming
static int mm2s(uint64_t bus_addr, uint32_t nbytes){
    uint32_t addr_lo = (uint32_t)(bus_addr & 0xFFFFFFFFULL);
    uint32_t addr_hi = (uint32_t)(bus_addr >> 32);

    REG_WR(dma_regs, MM2S_SA,     addr_lo);
    REG_WR(dma_regs, MM2S_SA_MSB, addr_hi);

    REG_WR(dma_regs, MM2S_DMACR,
           DMACR_RS | DMACR_IOC_IrqEn | DMACR_ERR_IrqEn);

    REG_WR(dma_regs, MM2S_LENGTH, nbytes);

    reg4[0] = 2; //command the PL master to start mm2s to read from DDR
    __sync_synchronize();
    reg4[0] = 0; //turn off right away to avoid triggering a second mm2s
    __sync_synchronize();
    return 0;
}

static int resetFSM(){
    reg4[0] = 3; //command the PL master to reset its FSM to be ready for a later s2mm
    __sync_synchronize();
    reg4[0] = 0; //turn off right away to avoid triggering a second mm2s
    __sync_synchronize();    
}

/* ------------------------------------------------------------------ */
/* 5. Poll DMASR until idle or error                                   */
/* ------------------------------------------------------------------ */
static int wait_s2mm(void)
{
    int timeout_ms = 5000;

    while (timeout_ms > 0) {
	
        uint32_t sr = REG_RD(dma_regs, S2MM_DMASR);
        if (sr & DMASR_ANY_ERR) {
            fprintf(stderr, "[dma] S2MM ERROR: DMASR=0x%08X\n", sr);
            if (sr & DMASR_DMAINTERR) fprintf(stderr, "  → Internal DMA error\n");
            if (sr & DMASR_DMASLVERR) fprintf(stderr, "  → Slave error\n");
            if (sr & DMASR_DMADECERR) fprintf(stderr, "  → Decode error\n");
            return -1;
        }

        if (sr & DMASR_ERR_Irq) {
            fprintf(stderr, "[dma] Error interrupt asserted: DMASR=0x%08X\n", sr);
            return -1;
        }

        /*
         * Idle bit (bit 1) is the authoritative "transfer complete" indicator
         * in Simple DMA (non-SG) polling mode.
         */
        if (sr & DMASR_IDLE) {
            printf("[dma] S2MM idle/complete: DMASR=0x%08X\n", sr);
            return 0;
        }

        usleep(1000); /* 1 ms poll interval */
    }
    fprintf(stderr, "[dma] S2MM wait timed out!\n");
    return -1;
}

/* ------------------------------------------------------------------ */
/* 5b. Poll MM2S DMASR until idle or error — mirror of wait_s2mm      */
/* ------------------------------------------------------------------ */
static int wait_mm2s(void)
{
    int timeout_ms = 5000;

    while (timeout_ms > 0) {
        uint32_t sr = REG_RD(dma_regs, MM2S_DMASR);
        //printf("in wait_s2mm, current sr = %x\n", sr);

        if (sr & DMASR_ANY_ERR) {
            fprintf(stderr, "[dma] MM2S ERROR: DMASR=0x%08X\n", sr);
            if (sr & DMASR_DMAINTERR) fprintf(stderr, "  → Internal DMA error\n");
            if (sr & DMASR_DMASLVERR) fprintf(stderr, "  → Slave error\n");
            if (sr & DMASR_DMADECERR) fprintf(stderr, "  → Decode error\n");
            return -1;
        }

        if (sr & DMASR_ERR_Irq) {
            fprintf(stderr, "[dma] MM2S Error interrupt asserted: DMASR=0x%08X\n", sr);
            return -1;
        }

        if (sr & DMASR_IDLE) {
            printf("[dma] MM2S idle/complete: DMASR=0x%08X\n", sr);
            return 0;
        }

        usleep(1000);
    }
    fprintf(stderr, "[dma] MM2S wait timed out!\n");
    return -1;
}

/* ------------------------------------------------------------------ */
/* 6. Cleanup                                                          */
/* ------------------------------------------------------------------ */
static void cleanup(void)
{
    if (rx_virt != MAP_FAILED) {
        munlock(rx_virt, TRANSFER_BYTES);
        munmap(rx_virt, TRANSFER_BYTES);
    }
    if (tx_virt != MAP_FAILED) {
        munlock(tx_virt, TRANSFER_BYTES);
        munmap(tx_virt, TRANSFER_BYTES);
    }
    if (dma_regs != MAP_FAILED)
        munmap((void *)dma_regs, DMA_MAP_SIZE);

    if (buf_fd     >= 0) close(buf_fd);
    if (heap_fd    >= 0) close(heap_fd);
    if (tx_buf_fd  >= 0) close(tx_buf_fd);
    if (tx_heap_fd >= 0) close(tx_heap_fd);
    if (devmem_fd  >= 0) close(devmem_fd);
    munmap((void *)reg1, MAP_SIZE);
    munmap((void *)reg2, MAP_SIZE);
    munmap((void *)reg3, MAP_SIZE);
    munmap((void *)reg4, MAP_SIZE);
    munmap((void *)reg6, MAP_SIZE);
    munmap((void *)reg7, MAP_SIZE);
    close(fd1);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    int nworkers = strtol(argv[1], NULL, 10);
    NUM_CONSTANTS = strtol(argv[2], NULL, 10);
    int rc = 0;

    usleep(200000);
    
    // 1. Open physical memory
    fd1 = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd1 < 0) {
        perror("open /dev/mem failed");
        return -1;
    }

    // 2. Map FPGA address space into user space
    reg1 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base1);

    reg2 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base2);

    reg3 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base3);
    reg4 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base4);
    reg6 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base6);
    reg7 = (uint32_t *)mmap(NULL, MAP_SIZE,
                            PROT_READ | PROT_WRITE,
                            MAP_SHARED,
                            fd1,
                            FSM_base7);                            
    
    
    if ((reg1 == MAP_FAILED)||(reg2 == MAP_FAILED)||(reg3 == MAP_FAILED)||(reg4 == MAP_FAILED)||(reg6 == MAP_FAILED)||(reg7 == MAP_FAILED)) {
        perror("mmap failed");
        close(fd1);
        return -1;
    }

    printf("Initializing...\n");

    if (init_dma_regs()  != 0) { rc = 1; goto out; }
    if (init_rx_buffer() != 0) { rc = 1; goto out; }
    if (init_tx_buffer() != 0) { rc = 1; goto out; }


    if (virt_to_phys(rx_virt, &rx_bus_addr) != 0) { rc = 1; goto out; }
    if (virt_to_phys(tx_virt, &tx_bus_addr) != 0) { rc = 1; goto out; }
    printf("[init] RX bus address: 0x%016llX\n",   (unsigned long long)rx_bus_addr);
    printf("[init] TX bus address: 0x%016llX\n\n", (unsigned long long)tx_bus_addr);

    // --- Sanity check: confirm simple DMA mode (SGIncld must be 0)
    {
        uint32_t sr_s2mm = REG_RD(dma_regs, S2MM_DMASR);
        uint32_t sr_mm2s = REG_RD(dma_regs, MM2S_DMASR);
        if ((sr_s2mm | sr_mm2s) & DMASR_SGINCLD) {
            fprintf(stderr,
                "[error] SGIncld is set — AXI DMA IP is in Scatter-Gather mode.\n"
                "        Uncheck 'Enable Scatter Gather Engine' in Vivado IP,\n"
                "        regenerate, and reprogram.\n");
            rc = 1;
            goto out;
        }
        printf("[init] Simple DMA mode confirmed (SGIncld=0) for both channels\n");
    }


    if (reset_mm2s() != 0) { rc = 1; goto out; }
    if (reset_s2mm() != 0) { rc = 1; goto out; }


    printf("Create FOL inference input ...");
    //now prepare FOL input
    unsigned long long seed = (unsigned long long)time(NULL);
    //lcg_seed(seed);
    lcg_seed(0);

    /* ---- initial facts ---- */
    HashSet *initial_facts = (HashSet *)malloc(sizeof(HashSet));//HashSet is just a struct of 3 pointers!
    hs_init(initial_facts, tx_virt, (void *)&reg5[0]);


    int rx_new_facts_count_temp = 0;
    HashSet *rx_new_facts = (HashSet *)malloc(sizeof(HashSet));
    rx_new_facts->count = &rx_new_facts_count_temp;//actually useless               
    rx_new_facts->slots = (Tuple *)((char *)rx_virt + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char));
    rx_new_facts->used  = (char *)rx_virt + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char) + HS_CAP*sizeof(Tuple);

    int new_facts_count_temp = 0;
    HashSet *new_facts = (HashSet *)malloc(sizeof(HashSet));//HashSet is just a struct of 3 pointers!
    hs_init_new(new_facts, tx_virt, (void *)(&new_facts_count_temp));//this new facts will be emptied by hs_init_new function
    rules_init(tx_virt, (void *)&reg5[2]);
    const_init((void *)&reg6[0]);
    active_init(tx_virt, (void *)&reg6[2]);
    for (int i = 0; i < NUM_CONSTANTS; i++) consts[i] = i;
    *nc = NUM_CONSTANTS;
     

    /* Unary facts */
    /* Full coverage for first FULL_COVERAGE_FIRST_N predicates */
    int lim = (FULL_COVERAGE_FIRST_N < NUM_UNARY) ? FULL_COVERAGE_FIRST_N : NUM_UNARY;
    for (int pred = 1; pred <= lim; pred++)
        for (int c = 0; c < *nc; c++)
            hs_insert(initial_facts, &(Tuple){{(uint16_t)pred, (uint16_t)consts[c], 0, 0}}, 2);

    /* Partial coverage for remaining unary predicates */
    int num_sel = (int)((*nc) * UNARY_DENSITY);
    if (num_sel < 1) num_sel = 1;
    int *sel = (int *)malloc(num_sel * sizeof(int));

    for (int pred = FULL_COVERAGE_FIRST_N + 1; pred <= NUM_UNARY; pred++) {
        lcg_sample(consts, *nc, num_sel, sel);
        for (int i = 0; i < num_sel; i++)
            hs_insert(initial_facts, &(Tuple){{(uint16_t)pred, (uint16_t)sel[i], 0, 0}}, 2);
    }
    free(sel);

    /* Hub constants: all predicates for NUM_HUB_CONSTANTS random consts */
    if (NUM_HUB_CONSTANTS > 0) {
        int nh = (NUM_HUB_CONSTANTS < *nc) ? NUM_HUB_CONSTANTS : *nc;
        int *hubs = (int *)malloc(nh * sizeof(int));
        lcg_sample(consts, *nc, nh, hubs);
        for (int pred = 1; pred <= NUM_UNARY; pred++)
            for (int i = 0; i < nh; i++)
                hs_insert(initial_facts, &(Tuple){{(uint16_t)pred, (uint16_t)hubs[i], 0, 0}}, 2);
        free(hubs);
    }

    /* UE facts: symmetric pairs (0, i, j) for i < j */
    for (int i = 0; i < *nc; i++)
        for (int j = i + 1; j < *nc; j++)
            hs_insert(initial_facts, &(Tuple){{0, (uint16_t)consts[i], (uint16_t)consts[j], 0}}, 3);

    /* ---- build rules ---- */
    /* Variables: VB=1024, VB+1=1025, VB+2=1026  (vb, vb2, vb3 in Python) */
    const int X = VB, Y = VB+1, Z = VB+2;

    /* Layer 1: UE(x,y) + P1(x) + P2(y) → Q(x,y)   argc=2 */
    for (int i = 0; i < RULES_LAYER1; i++) {
        int p1 = (i % NUM_UNARY) + 1;
        int p2 = ((i + 7) % NUM_UNARY) + 1;
        int q  = 50 + (i % NUM_BINARY);
        Tuple prems[3] = { make_tuple2(0,(uint16_t)p1,(uint16_t)p2), make_tuple1((uint16_t)p1,(uint16_t)X), make_tuple1((uint16_t)p2,(uint16_t)Y) };
        uint8_t arity_p[3] = {3, 2, 2}; 
        /* Fix: match Python's exact premise tuples */
        prems[0] = make_tuple2(0, (uint16_t)X, (uint16_t)Y);
        prems[1] = make_tuple1((uint16_t)p1, (uint16_t)X);
        prems[2] = make_tuple1((uint16_t)p2, (uint16_t)Y);
        Tuple concs[1] = { make_tuple2((uint16_t)q, (uint16_t)X, (uint16_t)Y) };
        uint8_t arity_c[1] = {3};
        //add_rule(2, 3, prems, 1, concs);
        add_rule(2, 3, prems, arity_p, 1, concs, arity_c);   
    }

    /* Layer 2: Q1(x,y) + P(x) + UE(x,z) → Q2(y,z)   argc=3 */
    for (int i = 0; i < RULES_LAYER2; i++) {
        int q1 = 50 + (i % NUM_BINARY);
        int p  = ((i + 3) % NUM_UNARY) + 1;
        int q2 = 50 + ((i + 5) % NUM_BINARY);
        Tuple prems[3] = {
            make_tuple2((uint16_t)q1, (uint16_t)X, (uint16_t)Y),
            make_tuple1((uint16_t)p,  (uint16_t)X),
            make_tuple2(0,  (uint16_t)X, (uint16_t)Z)
        };
        uint8_t arity_p[3] = {3, 2, 3};
        Tuple concs[1] = { make_tuple2((uint16_t)q2, (uint16_t)Y, (uint16_t)Z) };
        uint8_t arity_c[1] = {3};
        //add_rule(3, 3, prems, 1, concs);
        add_rule(3, 3, prems, arity_p, 1, concs, arity_c);
    }

    /* Layer 3: Q1(x,y) + Q2(y,z) + UE(x,z) → Q3(x,z)   argc=3 */
    for (int i = 0; i < RULES_LAYER3; i++) {
        int q1 = 50 + (i % NUM_BINARY);
        int q2 = 50 + ((i + 2) % NUM_BINARY);
        int q3 = 50 + ((i + 4) % NUM_BINARY);
        Tuple prems[3] = {
            make_tuple2((uint16_t)q1, (uint16_t)X, (uint16_t)Y),
            make_tuple2((uint16_t)q2, (uint16_t)Y, (uint16_t)Z),
            make_tuple2(0,  (uint16_t)X, (uint16_t)Z)
        };
        uint8_t arity_p[3] = {3, 3, 3};
        Tuple concs[1] = { make_tuple2((uint16_t)q3, (uint16_t)X, (uint16_t)Z) };
        uint8_t arity_c[1] = {3};
        add_rule(3, 3, prems, arity_p, 1, concs, arity_c);
    }

    /* Layer 4: P1(x) + UE(x,y) + P2(y) → P3(x)   argc=2 */
    for (int i = 0; i < RULES_LAYER4; i++) {
        int p1 = (i % NUM_UNARY) + 1;
        int p2 = ((i + 10) % NUM_UNARY) + 1;
        int p3 = ((i + 20) % NUM_UNARY) + 1;
        Tuple prems[3] = {
            make_tuple1((uint16_t)p1, (uint16_t)X),
            make_tuple2(0,  (uint16_t)X, (uint16_t)Y),
            make_tuple1((uint16_t)p2, (uint16_t)Y)
        };
        uint8_t arity_p[3] = {2, 3, 2};
        Tuple concs[1] = { make_tuple1((uint16_t)p3, (uint16_t)X) };
        uint8_t arity_c[1] = {2};
        //add_rule(2, 3, prems, 1, concs);
        add_rule(2, 3, prems, arity_p, 1, concs, arity_c);
    }

    /* Layer 5: Q(x,y) + P1(x) → P2(y)   argc=2 */
    for (int i = 0; i < RULES_LAYER5; i++) {
        int q  = 50 + (i % NUM_BINARY);
        int p1 = ((i + 5) % NUM_UNARY) + 1;
        int p2 = ((i + 15) % NUM_UNARY) + 1;
        Tuple prems[2] = {
            make_tuple2((uint16_t)q,  (uint16_t)X, (uint16_t)Y),
            make_tuple1((uint16_t)p1, (uint16_t)X)
        };
        uint8_t arity_p[2] = {3, 2};
        Tuple concs[1] = { make_tuple1((uint16_t)p2, (uint16_t)Y) };
        uint8_t arity_c[1] = {2};
        //add_rule(2, 2, prems, 1, concs);
        add_rule(2, 2, prems, arity_p, 1, concs, arity_c);
    }

    /* Layer 6: Q1(x,y) + Q2(y,z) + Q3(x,z) → Q4(x,y)   argc=3 */
    for (int i = 0; i < RULES_LAYER6; i++) {
        int q1 = 50 + (i % NUM_BINARY);
        int q2 = 50 + ((i + 1) % NUM_BINARY);
        int q3 = 50 + ((i + 2) % NUM_BINARY);
        int q4 = 50 + ((i + 3) % NUM_BINARY);
        Tuple prems[3] = {
            make_tuple2((uint16_t)q1, (uint16_t)X, (uint16_t)Y),
            make_tuple2((uint16_t)q2, (uint16_t)Y, (uint16_t)Z),
            make_tuple2((uint16_t)q3, (uint16_t)X, (uint16_t)Z)
        };
        uint8_t arity_p[3] = {3, 3, 3};
        Tuple concs[1] = { make_tuple2((uint16_t)q4, (uint16_t)X, (uint16_t)Y) };
        uint8_t arity_c[1] = {3};
        //add_rule(3, 3, prems, 1, concs);
        add_rule(3, 3, prems, arity_p, 1, concs, arity_c);
    }

    *n_active = *rule_count;
    for (int i = 0; i < *rule_count; i++) active[i] = (uint16_t)i; //all rules active in the first iteration



    uint64_t *hello = (uint64_t *)tx_virt;
    uint32_t cnt3 = 0;
    FILE *myfd = fopen("input.txt", "w");
 
    for(int i=0; i<TRANSFER_WORDS; i++){
        fprintf(myfd, "%d %lx\n", i, hello[i]);
        if(hello[i]!=0) cnt3++;
    }
    fclose(myfd);

    reg7[0] = nworkers; //notify the PL of nworkers. Set it between 1 and 255. Currently only support 1-24
    
    printf("\nInitial number of facts is %d\n", initial_facts->count[0]);
    printf(">>> Loop started \n\n");
    uint32_t myCnt = 0;

    double iter_times[20];
    //infinite loop to keep asking the PL for frames of 1024 words
    
    struct timespec t0, t1;
    struct timespec ts, te;
    struct timespec t2, t3;
    //clock_t t0 = clock();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while(1)
    {
	clock_gettime(CLOCK_MONOTONIC, &ts);

	printf("\n === Iteration %d === \n", myCnt);
	printf("Number of active rule %d\n", *n_active);
 
        //this is to guarantee cache coherence, do not delete or move this line's position!
        //this should always be used after PS writes into DMA's DDR. 
        __builtin___clear_cache((char *)tx_virt, (char *)tx_virt + TRANSFER_BYTES);

        __sync_synchronize();

        printf("mm2s in progress...\n");

        //--- Arm MM2S first: DMA reads from TX buffer 
        if (mm2s(tx_bus_addr, TRANSFER_BYTES) != 0) { rc = 1; goto out; }
        // --- Wait for MM2S channels to complete
        if (wait_mm2s()!= 0) { rc = 1; goto out; }
        clock_gettime(CLOCK_MONOTONIC, &t2);


	int prev_ai = -1;
	int curr_ai;

        while(reg3[2]==0) //wait until all PL workers are done, at which point the PL master will set its worker_zero signal, which is sensed by GPIO reg3[2]
        {
        }

        clock_gettime(CLOCK_MONOTONIC, &t3);

        resetFSM(); //reset the PL master's FSM
       

        printf("s2mm in progress...\n");

    	if (s2mm(rx_bus_addr, TRANSFER_BYTES) != 0) { rc = 1; goto out; }
    	if (wait_s2mm() != 0) { rc = 1; goto out; }
       
       

        printf("MERGE on PS in progress...\n\n");

        int nfcnt = merge_and_update(initial_facts, rx_new_facts, rules, *rule_count);

	clock_gettime(CLOCK_MONOTONIC, &te);


       	printf(">>>In iteration %d, PL found %d new facts\n", myCnt, nfcnt);
	printf(">>>In iteration %d, execution time of PS+PL is %lf\n", myCnt, (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec)/1000000000.0); 
       	printf(">>>In iteration %d, execution time of PL is %lf\n", myCnt, (t3.tv_sec - t2.tv_sec) + (t3.tv_nsec - t2.tv_nsec)/1000000000.0); 
        printf(">>>In iteration %d, execution time of PS is %lf\n", myCnt, (te.tv_sec - ts.tv_sec) + (te.tv_nsec - ts.tv_nsec)/1000000000.0-(t3.tv_sec - t2.tv_sec) - (t3.tv_nsec - t2.tv_nsec)/1000000000.0); 
        


	if(nfcnt==0) break;

        myCnt++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf(">>> Loop done \n");
   
    hello = (uint64_t *)tx_virt; 
    cnt3 = 0;
    myfd = fopen("res.txt", "w");
    for(int i=0; i<TRANSFER_WORDS; i++){
        fprintf(myfd, "%d %lx\n", i, hello[i]);
        if(hello[i]!=0) cnt3++;
    }
    printf("===== number of non-zero units after inference is %d\n", cnt3); 
    fclose(myfd);
    
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1000000000.0;

    printf("------ Total number of facts in KB: %d\n", initial_facts->count[0]);
    printf("------ Total number of rules in KB: %d\n", MAX_RULES);
    printf("Total time is %f seconds\n", elapsed);
    printf("Number of workers is %d\n", nworkers);
    printf("Initial number of rules is %d\n", MAX_RULES); 
    printf("Initial number of constants is %d\n", NUM_CONSTANTS);
out:
    cleanup();
    return rc;
}
