"""Simulate current vs new SL/TP on actual trades + post-exit analysis."""
import MetaTrader5 as mt5,pandas as pd,numpy as np
mt5.initialize()
rates=mt5.copy_rates_from_pos('BTCUSD',mt5.TIMEFRAME_M15,0,800)
df=pd.DataFrame(rates);df['ts']=pd.to_datetime(df['time'],unit='s')

trades=[
    ('2026-05-21 21:00','2026-05-21 23:30',77242.83,296.60,-1),
    ('2026-05-22 01:00','2026-05-22 01:30',77229.24,353.34,-1),
    ('2026-05-22 05:00','2026-05-22 09:00',77714.85,391.42,1),
]

def sim(entry_px,atr,d,ets,xts,sl_m,tp_m):
    et=pd.Timestamp(ets);xt=pd.Timestamp(xts)
    bars=df[(df['ts']>=et)&(df['ts']<=xt)]
    if len(bars)==0:return None
    sl=entry_px-d*sl_m*atr;tp=entry_px+d*tp_m*atr
    be_sl=entry_px-d*0.05*atr;be=False
    best=entry_px;csl=sl;res=None;mfe=0
    for _,b in bars.iterrows():
        if d==1:
            best=max(best,b['high']);mfe=max(mfe,(best-entry_px)/atr)
            if b['low']<=csl:res=('sl',(csl-entry_px)/atr);break
            if b['high']>=tp:res=('tp',tp_m);break
            if not be and (b['close']-entry_px)/atr>=0.5*sl_m:be=True;csl=be_sl
        else:
            best=min(best,b['low']);mfe=max(mfe,(entry_px-best)/atr)
            if b['high']>=csl:res=('sl',(entry_px-csl)/atr);break
            if b['low']<=tp:res=('tp',tp_m);break
            if not be and (entry_px-b['close'])/atr>=0.5*sl_m:be=True;csl=be_sl
    if res is None:
        last=bars.iloc[-1]['close']
        res=('held',(last-entry_px)/atr if d==1 else(entry_px-last)/atr)
    return res,mfe,len(bars)

print('='*70)
print('CONFIG COMPARISON — May 21-22 Trades')
print('='*70)
configs=[(1.0,2.5,'Current 1:2.5'),(0.5,1.2,'New 0.5:1.2')]
for i,(ets,xts,epx,atr,d) in enumerate(trades):
    ds='LONG' if d==1 else 'SHORT'
    print('\nTrade %d: %s %s @ %d ATR=%d' % (i+1,ets,ds,epx,atr))
    for sl,tp,label in configs:
        r=sim(epx,atr,d,ets,xts,sl,tp)
        if r is None:print('  %s: no bars'%label);continue
        res,mfe,nbars=r
        print('  %s: %s at %+.2fR | MFE=%+.2fR | %d bars | SL=%d TP=%d' % (
            label,res[0],res[1],mfe,nbars,sl*atr,tp*atr))

# Post-exit analysis
print('\n'+'='*70)
print('POST-EXIT: Did price keep moving in trade direction?')
print('='*70)
for i,(ets,xts,epx,atr,d) in enumerate(trades):
    xt=pd.Timestamp(xts)
    after=df[(df['ts']>xt)][:12]
    if len(after)==0:print('Trade %d: no data'%(i+1));continue
    ds='LONG' if d==1 else 'SHORT'
    sp=after.iloc[0]['close'];ep=after.iloc[-1]['close']
    mr=(ep-sp)/atr if d==1 else (sp-ep)/atr
    print('Trade %d post-exit (%s): %d bars, move=%+.2fR in trade dir'%(i+1,ds,len(after),mr))
    for j in range(min(4,len(after))):
        b=after.iloc[j];px=b['close'];r=(px-sp)/atr if d==1 else(sp-px)/atr
        print('  +%d: close=%d (%+.2fR)'%(j,px,r))
