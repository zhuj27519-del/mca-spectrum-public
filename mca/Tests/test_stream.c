#include "mca_stream.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static mca_stream_t stream;
static mca_histogram_t histogram,decoded;
static uint8_t received[MCA_FRAME_SIZE];
static size_t used,limit;
static int fail,blocked;
static volatile uint32_t generation;
static int mutate_generation;
void mca_service(void) { if(mutate_generation) { generation+=2U; mutate_generation=0; } }
static int write_partial(void *context,const uint8_t *data,size_t length)
{
    (void)context;
    if(fail) { return -1; }
    if(blocked) { return 0; }
    if(length>limit) { length=limit; }
    assert(used+length<=sizeof(received)); memcpy(received+used,data,length); used+=length;
    return (int)length;
}
static void drain(void)
{
    for(unsigned budget=0;stream.active && budget<10000U;++budget) { mca_stream_pump(&stream,write_partial,NULL); }
    assert(!stream.active && used==MCA_FRAME_SIZE);
}
int main(void)
{
    uint32_t sequence;
    mca_init(&histogram); histogram.duration_ms=1000;
    histogram.spectrum[400]=1234; generation=2;
    mca_stream_init(&stream);
    assert(!mca_stream_offer(&stream,&histogram,&generation,100));
    mca_stream_connection(&stream,true,0); generation=1;
    assert(!mca_stream_offer(&stream,&histogram,&generation,0));
    generation=0; assert(!mca_stream_offer(&stream,&histogram,&generation,0));
    generation=2; mutate_generation=1;
    assert(!mca_stream_offer(&stream,&histogram,&generation,0));
    assert(!stream.active);
    assert(mca_stream_offer(&stream,&histogram,&generation,0));
    blocked=1; mca_stream_pump(&stream,write_partial,NULL);
    assert(stream.active && stream.offset==0 && stream.blocked==1);
    assert(!mca_stream_offer(&stream,&histogram,&generation,100));
    histogram.spectrum[400]=9999; blocked=0; limit=17; drain();
    assert(mca_frame_decode(received,sizeof(received),&decoded,&sequence)==MCA_OK);
    assert(decoded.spectrum[400]==1234 && sequence==0);
    assert(!mca_stream_offer(&stream,&histogram,&generation,99));
    used=0; assert(mca_stream_offer(&stream,&histogram,&generation,100));
    limit=1; mca_stream_pump(&stream,write_partial,NULL); assert(used==2);
    fail=1; mca_stream_pump(&stream,write_partial,NULL);
    assert(!stream.active && !stream.connected && stream.errors==1);
    fail=0; used=0; limit=1460; mca_stream_connection(&stream,true,200); stream.sequence=UINT32_MAX;
    assert(mca_stream_offer(&stream,&histogram,&generation,200)); drain();
    assert(mca_frame_decode(received,sizeof(received),&decoded,&sequence)==MCA_OK && sequence==UINT32_MAX);
    assert(decoded.spectrum[400]==9999 && stream.sequence==0);
    used=0; mca_stream_connection(&stream,true,UINT32_MAX-20U);
    assert(mca_stream_offer(&stream,&histogram,&generation,UINT32_MAX-20U)); drain();
    assert(!mca_stream_offer(&stream,&histogram,&generation,50U));
    used=0; assert(mca_stream_offer(&stream,&histogram,&generation,80U)); drain();
    puts("PASS stream: disconnected/odd/changing-generation guards, stable snapshot, partial writes, backpressure, disconnect/reconnect, sequence/tick wrap");
    FILE *f=fopen("stream-frame.bin","wb"); assert(f); assert(fwrite(received,1,sizeof(received),f)==sizeof(received)); fclose(f);
    return 0;
}
