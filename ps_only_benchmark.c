//kv260 project: parallel FOL inference on PL
//potential use: parallel AG generation on PL

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
//#include <sys/mman.h>
//#include <sys/ioctl.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>


// FOL control parameters start
//#define NUM_CONSTANTS           50  //25  //50
int NUM_CONSTANTS;
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
#define VB  32768

// hash table capacity (# of facts)
#define HS_CAP 65536 //49152

#define MAX_ARITY       4           // pred_id 1 + up to 3 args = 4
//#define MAX_RULES       (RULES_LAYER1+RULES_LAYER2+RULES_LAYER3+RULES_LAYER4+RULES_LAYER5+RULES_LAYER6)
int MAX_RULES;
#define RULE_LIMIT      2048
#define MAX_PREMISES    4           // max number of predicates in premise of a rule
#define MAX_NEWCNT      1024        /* iteration log depth */

/* Max permutations: argc ≤ 3, NUM_CONSTANTS=40 → 40^3 = 64 000 */
//#define MAX_PERMS       150000
long MAX_PERMS;

static void  *rx_virt;   /* S2MM destination buffer */
static void  *tx_virt;   /* MM2S source buffer      */

uint32_t *reg1, *reg2, *reg3, *reg4, *reg5, *reg6, *reg7;
int nworkers=1;

/* ============================================================
   Simple LCG random number generator
   (reproduces Python random.seed / random.sample behaviour for
    uniform sampling without replacement)
   ============================================================ */
static unsigned long long lcg_state;

double time_to_merge = 0.0;
double time_on_for = 0.0;
double *time_iter;

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
    hs->slots = NULL; hs->used = NULL; hs->count = NULL;
}

/* Returns 1 if newly inserted, 0 if already present */
static int hs_insert(HashSet *hs, const Tuple *t, const char len) {
    unsigned int h = tuple_hash(t, len) % HS_CAP;
    while (hs->used[h]) {
        if (tuple_eq(&hs->slots[h], hs->used[h], t, len)) return 0;
        h = (h + 1) % HS_CAP;
    }
    //if(h==297) {
 //	    uint64_t *pt = (uint64_t *)(&(hs->slots[h]));
//	    printf("my length is %d\n", len);
//	    printf("Before insert, the unit is %lx\n", *pt);
  //  }
    hs->slots[h] = *t;
    hs->used[h]  = len; //used unit indicates both used and length of stored predicate
    hs->count[0]++;
    //if(h==297) {
    //        uint64_t *pt = (uint64_t *)(&(hs->slots[h]));
//	    printf("After insert, the unit is %lx\n", *pt);
  //  }
    return h;
}

static int hs_contains(const HashSet *hs, const Tuple *t, const char len) {
    unsigned int h = tuple_hash(t, len) % HS_CAP;
    while (hs->used[h]) {
        if (tuple_eq(&hs->slots[h], hs->used[h], t, len)) return 1;
        h = (h + 1) % HS_CAP;
    }
    return 0;
}

static void hs_merge(HashSet *dst, const HashSet *src) {
    for (int i = 0; i < HS_CAP; i++)
        if (src->used[i]) hs_insert(dst, &src->slots[i], src->used[i]);
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

/* ============================================================
   FOL_inference
   ============================================================ */
static void FOL_inference(int *consts, int nc,
                           Rule *rules, uint8_t *rule_nargs, int nrules,
                           HashSet *facts, HashSet *new_facts,
                           int vb,
                           int *newCnt_out, int *newCnt_sz,
                           int *ruleCoverage,  int *rc_sz, PermSet *perm_all)
{
    time_iter = (double *)malloc(10*sizeof(double));
    *n_active = nrules;
    for (int i = 0; i < nrules; i++) active[i] = (uint16_t)i;

    *newCnt_sz = 0;
    *rc_sz     = 0;

    char rc_set[MAX_RULES];
    memset(rc_set, 0, sizeof(rc_set));

    /* Preallocate permutation storage on the heap */
    //PermSet perms;
    //perms.data = (int (*)[MAX_ARITY-1])malloc(MAX_PERMS * sizeof(*perms.data));
    //if (!perms.data) { perror("perms alloc"); exit(1); }

    

    while (1) {
	printf(">>> In inference: iteration %d\n", *newCnt_sz);
        printf("Number of active rules is %d\n", *n_active);
        /* reset new_facts */
        memset((void *)(new_facts->used), 0, HS_CAP * sizeof(char));
        new_facts->count[0] = 0;

        int matched_cnt = 0;
        omp_set_num_threads(nworkers);
        #pragma omp parallel for
        for (int ai = 0; ai < *n_active; ai++) {
            int   r    = (int)active[ai];
            Rule *rule = &rules[r];
            uint8_t nargs = rule_nargs[r];
            int np3 = rule->premises.n&0x000000FF;
            int nc3 = rule->conclusions.n&0x000000FF;
            uint8_t *cnt_p = (uint8_t *)malloc(np3);
            uint8_t *cnt_c = (uint8_t *)malloc(nc3);
            uint64_t cp1 = rule->premises.n>>8;
            for(int i = 0; i < np3; i++){
                cnt_p[np3-1-i] = (uint8_t)(cp1&0x000000FF);
                cp1 = cp1 >> 8;
            }
            cp1 = rule->conclusions.n>>8;
            for(int i = 0; i < nc3; i++){
                cnt_c[nc3-1-i] = (uint8_t)(cp1&0x000000FF);
                cp1 = cp1 >> 8;
            }

	    PermSet *perms;
            if (nargs > 0) {
		if(nargs==1) perms = &perm_all[0];
		else if(nargs==2) perms = &perm_all[1];
		else if(nargs==3) perms = &perm_all[2];
                //make_perm(consts, nc, (int)nargs, &perms);

                for (int pi = 0; pi < perms->count; pi++) {
                    int *s = perms->data[pi];

                    /* check premises */
                    int matched = 1;
                    for (int pi2 = 0; pi2 < np3 && matched; pi2++) {
                        const Tuple *p = &rule->premises.clauses[pi2];
                        Tuple clause;
                        clause.v[0] = p->v[0]; //predicate name is copied directly
                        for (int k = 1; k < cnt_p[pi2]; k++) //each predicate argument is copied
                            clause.v[k] = (p->v[k] >= vb) ? s[p->v[k]-vb] : p->v[k];
                        //if (!hs_contains(facts, &clause)) { matched = 0; }
                        if(!hs_contains(facts, &clause, (char)cnt_p[pi2])) { matched = 0; }
                    }
                    if (!matched) continue;
                    //matched_cnt++;

                    /* derive conclusions */
                    for (int ci = 0; ci < nc3; ci++) {
                        const Tuple *cc = &rule->conclusions.clauses[ci];
                        Tuple clause;
                        clause.v[0] = cc->v[0];
                        for (int k = 1; k < MAX_ARITY; k++){
			    if(k<cnt_c[ci]) clause.v[k] = (cc->v[k] >= vb) ? s[cc->v[k]-vb] : cc->v[k];
			    else clause.v[k] = 0;
			}
                        if (!hs_contains(facts, &clause, (char)cnt_c[ci])) {
			    #pragma omp critical
			    {int hh = hs_insert(new_facts, &clause, (char)cnt_c[ci]);}
			    /*if(hh==297){
                                printf("The rule is %d\n", r);
			    }*/
                            rc_set[r] = 1;
                        }
                    }
                }
            } else {
                /* no-variable rule */
                int matched = 1;
                for (int pi2 = 0; pi2 < np3 && matched; pi2++)
                    if (!hs_contains(facts, &rule->premises.clauses[pi2], (char)cnt_p[pi2]))
                        matched = 0;
                if (!matched) continue;
                matched_cnt ++;
                for (int ci = 0; ci < nc3; ci++) {
                    const Tuple *cc = &rule->conclusions.clauses[ci];
                    if (!hs_contains(facts, cc, (char)cnt_c[ci])) {
			#pragma omp critical
			{int hh = hs_insert(new_facts, cc, (char)cnt_c[ci]);}
			/*if(hh==297){
                            printf("The no-var rule is %d\n", r);
			}*/
                        rc_set[r] = 1;
                    }
                }
            }
        } /* rule loop */

        /*if(*newCnt_sz == 0){
            FILE *myfd2 = fopen("imm_res.txt", "w");
            uint64_t *nfact = (uint64_t *)rx_virt;
            for(int k=0; k<180224; k++){
                fprintf(myfd2, "%d %16lx\n", k, nfact[k]);
            }
            fclose(myfd2);
            myfd2 = fopen("new_facts.txt", "w");
            nfact = (uint64_t *)new_facts->slots;
            for(int k=0; k<65536; k++){
                fprintf(myfd2, "%d %16lx\n", k, nfact[k]);
            }
            fclose(myfd2); 
            myfd2 = fopen("new_used.txt", "w");
            uint8_t *nused = (uint8_t *)new_facts->used;
            for(int k=0; k<65536; k++){
                fprintf(myfd2, "%d %x\n", k, nused[k]);
            }
            fclose(myfd2); 

        }*/
        printf("Iteration %d found %d facts\n", *newCnt_sz, new_facts->count[0]);


        if (new_facts->count[0] == 0) break;

        if (*newCnt_sz < MAX_NEWCNT)
            newCnt_out[(*newCnt_sz)++] = new_facts->count[0];


        hs_merge(facts, new_facts);


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
    }

    *rc_sz = 0;
    for (int r = 0; r < nrules; r++)
        if (rc_set[r]) ruleCoverage[(*rc_sz)++] = r;

    //free(perms.data);
    hs_free(new_facts);
}




/* ------------------------------------------------------------------ */
/* 6. Cleanup                                                          */
/* ------------------------------------------------------------------ */
static void cleanup(void)
{
    free(rx_virt);
    free(tx_virt);
    free(reg1);
    free(reg2);
    free(reg3);
    free(reg4);
    free(reg5);
    free(reg6);
    free(reg7);
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char *argv[])
{
    nworkers = strtol(argv[1], NULL, 10);

    int rc = 0;

    FILE *fp;
    char line[256], nextLine[256];
    char *token;
    
    fp = fopen(argv[2], "r");
    
    while(fgets(line, sizeof(line), fp)!=NULL){
        if(strncmp(line, "n_const", 7)==0){
            token = strtok(line, " \t\n");
            token = strtok(NULL, " \t\n");
            NUM_CONSTANTS = atoi(token);
        }
    }
    fclose(fp);
    printf("Number of constants in the input is %d\n", NUM_CONSTANTS);

    fp = fopen(argv[3], "r");
    
    while(fgets(line, sizeof(line), fp)!=NULL){
        if(strncmp(line, "n_rule", 6)==0){
            token = strtok(line, " \t\n");
            token = strtok(NULL, " \t\n");
            MAX_RULES = atoi(token);
        }
    }
    fclose(fp);
    printf("Number of rules in the input is %d\n", MAX_RULES);

    usleep(200000);
    
    reg1 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg2 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg3 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg4 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg5 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg6 = (uint32_t *)malloc(2*sizeof(uint32_t));
    reg7 = (uint32_t *)malloc(2*sizeof(uint32_t));
    tx_virt = (void *)((uint64_t *)malloc(262144*sizeof(uint64_t)));
    rx_virt = (void *)((uint64_t *)malloc(262144*sizeof(uint64_t)));                            

    //now prepare FOL input
    unsigned long long seed = (unsigned long long)time(NULL);
    //lcg_seed(seed);
    lcg_seed(0);

    /* ---- initial facts ---- */
    HashSet *initial_facts = (HashSet *)malloc(sizeof(HashSet));//HashSet is just a struct of 3 pointers!
    hs_init(initial_facts, tx_virt, (void *)&reg5[0]);
    HashSet *new_facts = (HashSet *)malloc(sizeof(HashSet));//HashSet is just a struct of 3 pointers!
    hs_init_new(new_facts, tx_virt, (void *)&reg7[0]);
    rules_init(tx_virt, (void *)&reg5[2]);
    const_init((void *)&reg6[0]);
    active_init(tx_virt, (void *)&reg6[2]);
    for (int i = 0; i < NUM_CONSTANTS; i++) consts[i] = i;
    *nc = NUM_CONSTANTS; 

    /* Unary facts */
    /* Full coverage for first FULL_COVERAGE_FIRST_N predicates */
    fp = fopen(argv[2], "r");
    
    while(fgets(line, sizeof(line), fp)!=NULL){
        if(strncmp(line, "n_const", 7)==0) break;
        uint16_t tokens[4] = {0, 0, 0, 0};
        int ti = 0;
        token = strtok(line, " \t\n");
        while(token!=NULL){
            tokens[ti] = (uint16_t)atoi(token);
            ti++;
            token = strtok(NULL, " \t\n");
        }
        hs_insert(initial_facts, &(Tuple){{tokens[0], tokens[1], tokens[2], tokens[3]}}, ti);
    }
    fclose(fp);

    fp = fopen(argv[3], "r");
    
    int rc5=0;
    while(fgets(line, sizeof(line), fp)!=NULL){
        if(strncmp(line, "n_rule", 6)==0) break;
        if(strncmp(line, "nargs", 5)==0){//start of a rule, line of rule title
	    rc5++;
            token = strtok(line, " \t\n");
            token = strtok(NULL, " \t\n");
            uint8_t my_argc = (uint8_t)atoi(token);
            fgets(nextLine, sizeof(nextLine), fp); //line of np
            token = strtok(nextLine, " \t\n");
            token = strtok(NULL, " \t\n");
            int my_np = atoi(token);
            fgets(nextLine, sizeof(nextLine), fp); //line of arity for premise predicates
            uint8_t *my_cnt_p = (uint8_t *)calloc(my_np, sizeof(uint8_t));
            int ii=0;
            token = strtok(nextLine, " \t\n");
            while(token!=NULL){
                my_cnt_p[ii] = (uint8_t)atoi(token);
                ii++;
                token = strtok(NULL, " \t\n");
            }
            Tuple *prems = (Tuple *)calloc(my_np, sizeof(Tuple));
            for(int ii=0; ii<my_np; ii++){
                fgets(nextLine, sizeof(nextLine), fp);//a new line of premise predicate
                int jj=0;
                token = strtok(nextLine, " \t\n");
                while(token!=NULL){
                    prems[ii].v[jj] = (uint16_t)atoi(token);
                    jj++;
                    token = strtok(NULL, " \t\n");
                }
            }
            fgets(nextLine, sizeof(nextLine), fp); //line of nc
            token = strtok(nextLine, " \t\n");
            token = strtok(NULL, " \t\n");
            int my_nc = atoi(token);
            fgets(nextLine, sizeof(nextLine), fp); //line of arity for conclusion predicates
            uint8_t *my_cnt_c = (uint8_t *)calloc(my_nc, sizeof(uint8_t));
            ii=0;
            token = strtok(nextLine, " \t\n");
            while(token!=NULL){
                my_cnt_c[ii] = (uint8_t)atoi(token);
                ii++;
                token = strtok(NULL, " \t\n");
            }
            Tuple *concs = (Tuple *)calloc(my_nc, sizeof(Tuple));
            for(int ii=0; ii<my_nc; ii++){
                fgets(nextLine, sizeof(nextLine), fp);//a new line of premise predicate
                int jj=0;
                token = strtok(nextLine, " \t\n");
                while(token!=NULL){
                    concs[ii].v[jj] = (uint16_t)atoi(token);
                    jj++;
                    token = strtok(NULL, " \t\n");
                }
            }
            add_rule(my_argc, my_np, prems, my_cnt_p, my_nc, concs, my_cnt_c);
            free(my_cnt_p);
            free(prems);
            free(my_cnt_c);
            free(concs);
            printf("Intialized rule %d\n", rc5);

        }
    }
    fclose(fp);

    memcpy(rx_virt, tx_virt, 262144*sizeof(uint64_t));

    //printf("Initial number of facts in KB %d\n", initial_facts->count[0]);

    printf(">>>Redirect all the pointers to rx buffer<<<\n");

    initial_facts->slots = (Tuple *)rx_virt;
    printf("------initial_facts slot offset %ld\n", 0L);
    initial_facts->used  = (char  *)rx_virt + HS_CAP*sizeof(Tuple);
    printf("------initial_facts used offset %ld\n", HS_CAP*sizeof(Tuple));

    new_facts->slots = (Tuple *)((char *)rx_virt + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char));
    printf("------new_facts slot offset %ld\n", HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char));
    new_facts->used  = (char *)rx_virt + HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char) + HS_CAP*sizeof(Tuple);
    printf("------new_facts used offset %ld\n", HS_CAP*sizeof(Tuple) + HS_CAP*sizeof(char) + HS_CAP*sizeof(Tuple));

    rules = (Rule *)((char *)rx_virt + 2*HS_CAP*sizeof(Tuple) + 2*HS_CAP*sizeof(char));
    printf("------rules offset %ld\n", 2*HS_CAP*sizeof(Tuple) + 2*HS_CAP*sizeof(char)); 
    rule_nargs = (uint8_t *)((char *)rx_virt + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule));
    printf("------rule_nargs offset %ld\n", HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule));

    active = (uint16_t *)((char *)rx_virt + HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule)+ 32768);
    printf("------active offset %ld\n", HS_CAP*sizeof(Tuple)*2 + HS_CAP*sizeof(char)*2 + RULE_LIMIT*sizeof(Rule)+ 32768);  
    

    /*uint64_t *hello = (uint64_t *)rx_virt;
    uint32_t cnt3 = 0;
    FILE *myfd = fopen("res.txt", "w");
 
    for(int i=0; i<262144; i++){
	    fprintf(myfd, "%d %lx\n", i, hello[i]);
	    if(hello[i]!=0) cnt3++;
    }

    printf("===== number of non-zero units is %d\n", cnt3); 
    fclose(myfd);*/
    
    printf(">>>Redirection is done<<<\n");
    printf("Initial number of facts in KB %d\n", initial_facts->count[0]);

    int newCnt[MAX_NEWCNT];
    int newCnt_sz = 0;
    int ruleCoverage[MAX_RULES];
    int rc_sz = 0;


    //typedef struct {
        //int (*data)[MAX_ARITY-1];   // heap: [MAX_PERMS][MAX_ARITY-1] 
        //int count;
    //} PermSet;


    //PermSet perms;
    //perms.data = (int (*)[MAX_ARITY-1])malloc(MAX_PERMS * sizeof(*perms.data));
    //if (!perms.data) { perror("perms alloc"); exit(1); }

    PermSet perms1;

    PermSet perm_all[MAX_ARITY-2];

    MAX_PERMS = (long)NUM_CONSTANTS * (long)NUM_CONSTANTS;

    for(int i=0; i<MAX_ARITY-2; i++){
       perm_all[i].data = (int (*)[MAX_ARITY-1])malloc(MAX_PERMS * sizeof(*perms1.data));
       make_perm(consts, *nc, i+1, &perm_all[i]);
    }





    printf("------ Inference started\n");
    
    struct timespec t0, t1;
    //clock_t t0 = clock();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    //FOL_inference(consts, *nc, rules, *rule_count, initial_facts, VB, newCnt, &newCnt_sz, ruleCoverage, &rc_sz);
    FOL_inference(consts, *nc, rules, rule_nargs, *rule_count, initial_facts, new_facts, VB, newCnt, &newCnt_sz, ruleCoverage, &rc_sz, perm_all);

    //clock_t t1 = clock();
    //gettimeofday(&t1, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    
    //double elapsed = (double)(t1 - t0) / CLOCKS_PER_SEC;
    double elapsed = (t1.tv_sec-t0.tv_sec) + (t1.tv_nsec-t0.tv_nsec)/1000000000.0;
    printf("------ Total number of rules in KB: %d\n", *rule_count);
    printf("------ Number of activated rules: %d\n", rc_sz);
    printf("------ Total number of facts in KB: %d\n", initial_facts->count[0]);
    printf("------ Total number of constants in KB: %d\n", *nc);
    printf("Total time is %f seconds\n", elapsed);
    printf("Time to merge fact tables is %f seconds\n", time_to_merge);
    printf("Time to infer new facts is %f seconds\n", time_on_for);
    //printf("=== Extra info ===");
    //for(int i=0; i<newCnt_sz; i++){
    //    printf("iteration %d found %d new facts\n", i, newCnt[i]);
    //    printf("inference of these new facts took %f seconds\n", time_iter[i]);
    //}

out:
    free(perm_all[0].data);
    free(perm_all[1].data);
    //free(perm_all[2].data);
    cleanup();
    return rc;
}
