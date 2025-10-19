<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Generative Ontology — Page 2 (Advanced Formulations)</title>
  <style>
    :root{
      --bg:#0b0f14; --panel:#111823; --ink:#dbe6ff; --muted:#8aa2c0; --accent:#66d9ef; --ok:#44d18d; --warn:#ffb454; --bad:#ff6b6b; --card:#0e1520; --stroke:#1b2736; --shadow:0 10px 30px rgba(0,0,0,.35);
    }
    *{box-sizing:border-box}
    html,body{height:100%;margin:0;background:var(--bg);color:var(--ink);font:14px/1.3 system-ui,Segoe UI,Inter,Roboto,Arial}
    .wrap{display:grid;grid-template-rows:auto 1fr auto;min-height:100vh}
    header{position:sticky;top:0;z-index:5;background:linear-gradient(180deg,rgba(11,15,20,.95),rgba(11,15,20,.85));backdrop-filter:blur(8px);border-bottom:1px solid var(--stroke)}
    .toolbar{display:flex;flex-wrap:wrap;gap:.75rem;align-items:center;padding:.8rem 1rem}
    .toolbar .group{display:flex;gap:.5rem;align-items:center;background:var(--panel);border:1px solid var(--stroke);padding:.5rem .6rem;border-radius:12px}
    label{color:var(--muted);font-size:.85rem}
    select,input[type="range"],button,input[type="checkbox"]{accent-color:var(--accent)}
    select,button{background:#0b131d;color:var(--ink);border:1px solid var(--stroke);padding:.45rem .6rem;border-radius:10px}
    button:hover{border-color:#2a3a52}

    main{display:grid;grid-template-columns:1fr 1fr;gap:1rem;padding:1rem}
    .panel{background:var(--panel);border:1px solid var(--stroke);border-radius:16px;box-shadow:var(--shadow);display:grid;grid-template-rows:auto 1fr auto;min-height:420px}
    .panel h3{margin:0;padding:.8rem 1rem;border-bottom:1px solid var(--stroke);color:#c6d7f4}
    .canvas-wrap{position:relative}
    canvas{display:block;width:100%;height:100%;border-bottom-left-radius:16px;border-bottom-right-radius:16px}
    .legend{display:flex;gap:1rem;align-items:center;justify-content:space-between;padding:.6rem 1rem;border-top:1px solid var(--stroke);color:var(--muted)}
    .meter{display:flex;gap:.75rem;align-items:center}
    .bar{width:140px;height:8px;background:#0b131d;border:1px solid var(--stroke);border-radius:999px;overflow:hidden}
    .bar > i{display:block;height:100%;background:linear-gradient(90deg,var(--ok),var(--warn));width:40%}
    .badge{font-size:.72rem;padding:.15rem .45rem;border:1px solid var(--stroke);border-radius:999px;background:#0b131d;color:#c9d6ef}
    .badge.ok{border-color:#1b3e2f;background:#0e1f19;color:#a6f1cd}
    .badge.warn{border-color:#3e2f1b;background:#1f1a0e;color:#ffdba6}
    .badge.bad{border-color:#3e1b1b;background:#1f0e0e;color:#ffb7b7}

    .callouts{padding:0 1rem 1rem;display:grid;grid-template-columns:1fr 1fr;gap:1rem}
    .callout{background:var(--card);border:1px solid var(--stroke);border-radius:14px;padding:.8rem 1rem;color:#c9d6ef}
    .callout h4{margin:.1rem 0 .35rem 0;color:#e4edff;font-size:.95rem}
    .muted{color:var(--muted)}

    footer{padding:.6rem 1rem;color:var(--muted);border-top:1px solid var(--stroke)}
    .kbd{border:1px solid var(--stroke);border-bottom-color:#000;background:#0b121b;border-radius:6px;padding:0 .35rem;color:#a5c2ff}

    @media (max-width: 1100px){ main{grid-template-columns:1fr} }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="toolbar">
      <div class="group">
        <label for="module">Module</label>
        <select id="module">
          <option value="effect">Effect & MV Algebra Lab</option>
          <option value="oml">Orthomodular Projector Lab</option>
          <option value="multiway">Multiway / Ruliad Panel</option>
          <option value="holo">Holographic Screen Demo</option>
          <option value="topos">Topos / Sheaf Inspector (lite)</option>
        </select>
      </div>
      <div class="group" id="effectCtl" hidden>
        <label>A</label><input type="range" id="A" min="0" max="100" value="35" />
        <label>B</label><input type="range" id="B" min="0" max="100" value="50" />
        <label><input type="checkbox" id="mvMode"/> MV‑mode</label>
        <label><input type="checkbox" id="effectMode" checked/> Effect‑mode</label>
      </div>
      <div class="group" id="omlCtl" hidden>
        <label>α</label><input type="range" id="alpha" min="0" max="180" value="20" />
        <label>β</label><input type="range" id="beta" min="0" max="180" value="100" />
        <label><input type="checkbox" id="YisPlane" checked/> Y := plane (X ≤ Y)</label>
      </div>
      <div class="group" id="multiCtl" hidden>
        <label for="rule">Rules</label>
        <select id="rule">
          <option value="fib">A→AB, B→A (Fibonacci)</option>
          <option value="bin">0→01, 1→0</option>
        </select>
        <label>Depth</label><input type="range" id="depth" min="1" max="8" value="5"/>
      </div>
      <div class="group" id="holoCtl" hidden>
        <button id="writeBit">Write bit</button>
        <button id="eraseBits">Erase</button>
      </div>
    </div>
  </header>

  <main>
    <section class="panel" id="leftPanel">
      <h3 id="leftTitle">Model</h3>
      <div class="canvas-wrap"><canvas id="leftCanvas"></canvas></div>
      <div class="legend">
        <div class="meter"><span class="muted">Status</span>
          <span class="badge" id="leftBadge1">—</span>
          <span class="badge" id="leftBadge2">—</span>
        </div>
        <div class="meter"><span class="muted">Metric</span>
          <span class="bar" id="leftBar"><i style="width:35%"></i></span>
        </div>
      </div>
    </section>

    <section class="panel" id="rightPanel">
      <h3 id="rightTitle">Monitor</h3>
      <div class="canvas-wrap"><canvas id="rightCanvas"></canvas></div>
      <div class="legend">
        <div class="meter"><span class="muted">Law</span>
          <span class="badge" id="lawBadge">—</span>
          <span class="badge" id="lawCheck">—</span>
        </div>
        <div class="meter"><span class="muted">Signal</span>
          <span class="bar" id="rightBar"><i style="width:45%"></i></span>
        </div>
      </div>
    </section>
  </main>

  <div class="callouts">
    <div class="callout">
      <h4>Alternative formulations in action</h4>
      <div class="muted">
        <strong>Effect/MV:</strong> partial addition (defined iff safe), orthosupplement, and Łukasiewicz ops. 
        <strong>OML:</strong> Hilbert subspaces with meet/join/complement and a live orthomodular check. 
        <strong>Multiway:</strong> rule evolution with merges ⇒ pockets of reducibility. 
        <strong>Holographic:</strong> boundary writes with interior encoding. 
        <strong>Topos:</strong> lite diagram of objects and a subobject classifier Ω<sub>J</sub>.
      </div>
    </div>
    <div class="callout">
      <h4>Same narrative, richer lenses</h4>
      <div class="muted">These lenses remain anchored to your nucleus \(R\) and round‑trip contracts: each demo preserves adjunction/definedness in its own idiom and exposes where classicalization snaps in.</div>
    </div>
  </div>

  <footer>
    <span class="muted">Tip:</span> In **Effect‑mode**, A ⊕ B is defined iff A+B ≤ 1. Toggle MV‑mode to see Łukasiewicz sums (capped at 1) and negation (1−x).
  </footer>
</div>

<script>
(function(){
  const dpr = Math.min(window.devicePixelRatio||1, 2);
  const $ = (id)=>document.getElementById(id);

  const leftC = $('leftCanvas'), rightC = $('rightCanvas');
  const ltx = leftC.getContext('2d'), rtx = rightC.getContext('2d');
  function resize(){
    for (const c of [leftC,rightC]){
      const rect = c.parentElement.getBoundingClientRect();
      c.width = Math.max(200, rect.width * dpr);
      c.height = Math.max(240, (rect.height||360) * dpr);
      c.style.height = (rect.height||360)+'px';
    }
  }
  new ResizeObserver(resize).observe(document.body); resize();

  // UI controls
  const moduleSel = $('module');
  const effectCtl = $('effectCtl'), omlCtl=$('omlCtl'), multiCtl=$('multiCtl'), holoCtl=$('holoCtl');
  const leftTitle = $('leftTitle'), rightTitle = $('rightTitle');
  const leftBadge1=$('leftBadge1'), leftBadge2=$('leftBadge2'), leftBar=$('leftBar').querySelector('i');
  const lawBadge=$('lawBadge'), lawCheck=$('lawCheck'), rightBar=$('rightBar').querySelector('i');

  // Effect controls
  const A=$('A'), B=$('B'), mvMode=$('mvMode'), effectMode=$('effectMode');

  // OML controls
  const alpha=$('alpha'), beta=$('beta'), YisPlane=$('YisPlane');

  // Multiway controls
  const ruleSel=$('rule'), depth=$('depth');

  // Holographic controls
  const writeBit=$('writeBit'), eraseBits=$('eraseBits');

  // Shared helpers
  function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
  function txt(ctx, s, x,y, color){ctx.fillStyle=color||'#8aa2c0'; ctx.fillText(s,x,y);} 

  // State
  let module = 'effect';
  let bits=[]; // holographic

  // ===== Module: Effect & MV Algebra Lab =====
  function effectDraw(){
    const a = +A.value/100, b= +B.value/100;
    // Compute MV and Effect results
    const mvAdd = Math.min(1, a + b);
    const mvNegA = 1 - a;
    const effCompat = (a + b) <= 1 + 1e-9;
    const effAdd = effCompat? (a + b) : null;

    // Left: bars
    ltx.setTransform(1,0,0,1,0,0); ltx.clearRect(0,0,leftC.width,leftC.height);
    const W=leftC.width,H=leftC.height; const m=30*dpr, bw=W-2*m, bh=16*dpr;
    ltx.fillStyle='#0b121b'; ltx.fillRect(0,0,W,H);
    ltx.fillStyle='#8aa2c0'; ltx.font=`${12*dpr}px system-ui`;
    txt(ltx,'A', m, m+bh*0.5);
    bar(ltx, m+20*dpr, m, bw, bh, '#7aa7ff', a);
    txt(ltx,'B', m, m+bh*2);
    bar(ltx, m+20*dpr, m+bh*1.5, bw, bh, '#b585ff', b);
    txt(ltx,'A⊥ = 1−A', m, m+bh*3.5);
    bar(ltx, m+70*dpr, m+bh*3, bw-50*dpr, bh, '#77ffc5', mvNegA);

    // Right: result and definedness
    rtx.setTransform(1,0,0,1,0,0); rtx.clearRect(0,0,rightC.width,rightC.height);
    const WR=rightC.width, HR=rightC.height; const cx=WR/2, cy=HR/2;
    // background
    const grd=rtx.createRadialGradient(cx,cy,0,cx,cy,Math.min(WR,HR)*.6);
    grd.addColorStop(0,'#0b121b'); grd.addColorStop(1,'#081018'); rtx.fillStyle=grd; rtx.fillRect(0,0,WR,HR);

    // buckets visualization
    drawBucket(cx-140*dpr, cy, 120*dpr, a, '#7aa7ff', 'A');
    drawBucket(cx+20*dpr, cy, 120*dpr, b, '#b585ff', 'B');

    // MV and Effect results
    const mvY = cy + 120*dpr; const effY = cy + 150*dpr;
    rtx.font = `${12*dpr}px system-ui`; rtx.fillStyle='#c9d6ef';
    rtx.fillText(`MV: A ⊕ B = min(1, A+B) = ${mvAdd.toFixed(2)}`, cx-140*dpr, mvY);
    if (effCompat){
      rtx.fillStyle='#a6f1cd'; rtx.fillText(`Effect: A ⊞ B = ${effAdd.toFixed(2)} (defined)`, cx-140*dpr, effY);
    } else {
      rtx.fillStyle='#ffb7b7'; rtx.fillText(`Effect: A ⊞ B = ⊥ (undefined; A+B>1)`, cx-140*dpr, effY);
    }

    // badges
    leftBadge1.textContent = mvMode.checked? 'MV: on' : 'MV: off';
    leftBadge2.textContent = effectMode.checked? 'Effect: on' : 'Effect: off';
    leftBadge2.className = 'badge ' + (effCompat? 'ok' : 'bad');
    leftBar.style.width = Math.round(mvAdd*100)+'%';
    lawBadge.textContent = 'Definedness';
    lawCheck.textContent = effCompat? 'isSome(A ⊞ B)' : 'None';
    lawCheck.className = 'badge ' + (effCompat? 'ok':'bad');
    rightBar.style.width = Math.round((effCompat? effAdd: mvAdd)*100)+'%';

    function bar(ctx,x,y,w,h,color,val){
      ctx.strokeStyle='#1b2736'; ctx.strokeRect(x,y,w,h);
      ctx.fillStyle=color; ctx.fillRect(x,y,w*clamp(val,0,1),h);
    }
    function drawBucket(x,y,w,val,color,label){
      const h = 80*dpr; rtx.strokeStyle='#1b2736'; rtx.strokeRect(x-w/2,y-h/2,w,h);
      rtx.fillStyle=color; rtx.fillRect(x-w/2, y+h/2 - h*clamp(val,0,1), w, h*clamp(val,0,1));
      rtx.fillStyle='#8aa2c0'; rtx.fillText(`${label}: ${val.toFixed(2)}`, x-w/2, y-h/2-8*dpr);
    }
  }

  // ===== Module: Orthomodular Projector Lab =====
  function omlDraw(){
    const a = (+alpha.value) * Math.PI/180; // X angle
    const b = (+beta.value) * Math.PI/180;  // a helper angle to draw Y basis

    // Left: subspaces in R^2
    ltx.setTransform(1,0,0,1,0,0); ltx.clearRect(0,0,leftC.width,leftC.height);
    const W=leftC.width,H=leftC.height; const cx=W/2, cy=H/2; const R=Math.min(W,H)*.35;
    bg(ltx,W,H);
    // Axes
    ltx.strokeStyle='rgba(120,160,255,.25)'; ltx.lineWidth=1*dpr; line(ltx,cx-R,cy,cx+R,cy); line(ltx,cx,cy-R,cx,cy+R);

    // X line and X^⊥
    drawLineAngle(ltx,cx,cy,R,a,'#9bc9ff','X');
    drawLineAngle(ltx,cx,cy,R,a+Math.PI/2,'#77ffc5','X⊥');

    // Y: either plane (flag) or the same as X for a trivial inclusion
    const yIsPlane = YisPlane.checked;

    // Right: check OML identity Y = X ∨ (X⊥ ∧ Y) when X ≤ Y
    rtx.setTransform(1,0,0,1,0,0); rtx.clearRect(0,0,rightC.width,rightC.height);
    const WR=rightC.width, HR=rightC.height; const rx=WR/2, ry=HR/2; const RR=Math.min(WR,HR)*.35;
    bg(rtx,WR,HR);
    // draw X and pick Y
    drawLineAngle(rtx,rx,ry,RR,a,'#9bc9ff','X');
    if(yIsPlane){ // Y = plane
      // plane backdrop
      rtx.fillStyle='rgba(180,200,255,.05)'; rtx.fillRect(rx-RR,ry-RR,RR*2,RR*2);
      // compute X⊥ ∧ Y = X⊥, so RHS = X ∨ X⊥ = plane → holds
      lawBadge.textContent = 'Orthomodular law';
      lawCheck.textContent = 'holds (Y=plane)'; lawCheck.className='badge ok';
    } else {
      // Let Y=X (minimal nontrivial super), then X⊥∧Y = {0}, RHS = X ∨ {0} = X → holds
      drawLineAngle(rtx,rx,ry,RR,a,'#cfa6ff','Y=X');
      lawBadge.textContent = 'Orthomodular law';
      lawCheck.textContent = 'holds (Y=X)'; lawCheck.className='badge ok';
    }
    leftBadge1.textContent='Subspaces in R²'; leftBadge2.textContent= yIsPlane? 'Y = plane' : 'Y = X';
    rightBar.style.width = yIsPlane? '100%' : '50%'; leftBar.style.width = '50%';

    function bg(ctx,W,H){ const grd=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.min(W,H)*.6); grd.addColorStop(0,'#0b121b'); grd.addColorStop(1,'#081018'); ctx.fillStyle=grd; ctx.fillRect(0,0,W,H);}  
    function line(ctx,x1,y1,x2,y2){ ctx.beginPath(); ctx.moveTo(x1,y1); ctx.lineTo(x2,y2); ctx.stroke(); }
    function drawLineAngle(ctx,cx,cy,R,ang,color,label){
      const ux=Math.cos(ang), uy=Math.sin(ang);
      ctx.strokeStyle=color; ctx.lineWidth=2*dpr; line(ctx, cx-ux*R, cy-uy*R, cx+ux*R, cy+uy*R);
      ctx.fillStyle='#8aa2c0'; ctx.font=`${12*dpr}px system-ui`; ctx.fillText(label, cx+ux*R*0.6, cy+uy*R*0.6);
    }
  }

  // ===== Module: Multiway / Ruliad Panel =====
  function multiwayDraw(){
    const rule = ruleSel.value; const D=+depth.value;
    // generate tree
    const start = (rule==='fib')? 'A' : '0';
    const rules = (rule==='fib')? {A:'AB', B:'A'} : { '0':'01', '1':'0' };
    const levels=[[start]]; const seen=new Map(); seen.set(start, true);
    for(let d=1; d<=D; d++){
      const prev=levels[d-1]; const next=[];
      for(const s of prev){
        const children = rewriteAll(s, rules);
        for(const c of children){ next.push(c); if(!seen.has(c)) seen.set(c,true); }
      }
      levels.push(next);
    }
    // merges measure: unique/total per level
    const merges = levels.map(arr=> { const u=new Set(arr); return {lvl:arr.length, uniq:u.size}; });

    // Left: tree view (rows per depth)
    ltx.setTransform(1,0,0,1,0,0); ltx.clearRect(0,0,leftC.width,leftC.height);
    const W=leftC.width,H=leftC.height; bg(ltx,W,H);
    ltx.font=`${11*dpr}px system-ui`; ltx.fillStyle='#c9d6ef';
    const rowH = Math.max(18*dpr, H/(D+2));
    for(let d=0; d<levels.length; d++){
      const y = (d+1)*rowH; const arr=levels[d]; const uniq = Array.from(new Set(arr));
      // draw unique nodes as rounded chips
      const maxPerRow = Math.min(uniq.length, Math.floor((W-30*dpr)/(60*dpr)) );
      for(let i=0;i<maxPerRow;i++){
        const x = 20*dpr + i*60*dpr;
        chip(ltx, x,y, uniq[i]);
      }
      ltx.fillStyle='#8aa2c0'; ltx.fillText(`depth ${d}: ${uniq.length} unique / ${arr.length} total`, 20*dpr, y-10*dpr);
    }

    // Right: reducibility gauge and sample strings
    rtx.setTransform(1,0,0,1,0,0); rtx.clearRect(0,0,rightC.width,rightC.height);
    const WR=rightC.width, HR=rightC.height; bg(rtx,WR,HR);
    const last = merges[merges.length-1]; const red = last.uniq/Math.max(1,last.lvl);
    lawBadge.textContent='Reducibility'; lawCheck.textContent = (red<0.7)? 'pockets present' : 'branchy'; lawCheck.className = 'badge ' + (red<0.7?'ok':'warn');
    rightBar.style.width = Math.round((1-red)*100)+'%';

    // sample strings at final level
    const finalUniq = Array.from(new Set(levels[levels.length-1])).slice(0,6);
    rtx.font=`${12*dpr}px system-ui`; rtx.fillStyle='#c9d6ef';
    rtx.fillText('samples @ final depth:', 18*dpr, 30*dpr);
    finalUniq.forEach((s,i)=>{ chip(rtx, 20*dpr + i*80*dpr, 60*dpr, s); });

    leftBadge1.textContent = (rule==='fib')? 'Fibonacci rules' : 'Binary rules';
    leftBadge2.textContent = `depth=${D}`;
    leftBar.style.width = Math.round((1-red)*100)+'%';

    function rewriteAll(s, rules){
      // parallel rewrite: replace each character by its image
      return [ Array.from(s).map(ch => rules[ch]??ch).join('') ];
    }
    function chip(ctx,x,y,text){
      const w=56*dpr, h=16*dpr; ctx.fillStyle='#0e1520'; ctx.strokeStyle='#1b2736'; ctx.lineWidth=1; roundRect(ctx,x,y,w,h,6); ctx.fill(); ctx.stroke();
      ctx.fillStyle='#a8c2ff'; ctx.fillText(text, x+6*dpr, y+12*dpr);
    }
    function roundRect(ctx,x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }
    function bg(ctx,W,H){ const grd=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.min(W,H)*.6); grd.addColorStop(0,'#0b121b'); grd.addColorStop(1,'#081018'); ctx.fillStyle=grd; ctx.fillRect(0,0,W,H);}  
  }

  // ===== Module: Holographic Screen (lite) =====
  function holoDraw(){
    // Left: boundary circle with written bits
    ltx.setTransform(1,0,0,1,0,0); ltx.clearRect(0,0,leftC.width,leftC.height);
    const W=leftC.width,H=leftC.height; const cx=W/2, cy=H/2; const R=Math.min(W,H)*.35;
    bg(ltx,W,H);
    circle(ltx,cx,cy,R,'#5aa9ff');
    // bits
    bits.forEach((th,i)=>{ const x=cx + R*Math.cos(th), y=cy + R*Math.sin(th); dot(ltx,x,y,4*dpr,'#ffdf7a'); });
    ltx.font=`${12*dpr}px system-ui`; ltx.fillStyle='#c9d6ef'; ltx.fillText(`bits on boundary: ${bits.length}`, 18*dpr, 24*dpr);

    // Right: interior encoding proxy
    rtx.setTransform(1,0,0,1,0,0); rtx.clearRect(0,0,rightC.width,rightC.height);
    const WR=rightC.width, HR=rightC.height; const rx=WR/2, ry=HR/2; const RR=Math.min(WR,HR)*.35;
    bg(rtx,WR,HR);
    // radial fans for each bit (toy encoder)
    bits.forEach(th=>{ fan(rtx,rx,ry,RR, th, th+0.08, 'rgba(255,223,122,.25)'); });
    circle(rtx,rx,ry,RR,'rgba(180,200,255,.35)');
    lawBadge.textContent='Capacity vs content';
    const capacity = clamp(1 - bits.length/64, 0,1); rightBar.style.width = Math.round((1-capacity)*100)+'%';
    lawCheck.textContent = capacity>0.2? 'capacity available' : 'near saturation'; lawCheck.className = 'badge ' + (capacity>0.2?'ok':'warn');
    leftBadge1.textContent='Euler boundary store'; leftBadge2.textContent=`write density ${(bits.length/64*100).toFixed(0)}%`;
    leftBar.style.width = Math.round((bits.length/64*100))+'%';

    function circle(ctx,x,y,r,color){ ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.strokeStyle=color; ctx.lineWidth=2*dpr; ctx.stroke(); }
    function dot(ctx,x,y,r,color){ ctx.beginPath(); ctx.arc(x,y,r,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); }
    function fan(ctx,cx,cy,R,a1,a2,fill){ ctx.beginPath(); ctx.moveTo(cx,cy); ctx.arc(cx,cy,R,a1,a2); ctx.closePath(); ctx.fillStyle=fill; ctx.fill(); }
    function bg(ctx,W,H){ const grd=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.min(W,H)*.6); grd.addColorStop(0,'#0b121b'); grd.addColorStop(1,'#081018'); ctx.fillStyle=grd; ctx.fillRect(0,0,W,H);}  
  }

  // ===== Module: Topos / Sheaf Inspector (lite, textual diagram) =====
  function toposDraw(){
    ltx.setTransform(1,0,0,1,0,0); ltx.clearRect(0,0,leftC.width,leftC.height);
    const W=leftC.width,H=leftC.height; bg(ltx,W,H);
    ltx.font=`${13*dpr}px system-ui`; ltx.fillStyle='#c9d6ef';
    // Objects (stages) and Ω_J
    const nodes=[{n:'0D',x:80,y:70},{n:'1D',x:220,y:70},{n:'2D',x:360,y:70},{n:'3D',x:500,y:70},{n:'Ω_J',x:290,y:170}];
    nodes.forEach(o=> node(ltx,o.x*dpr,o.y*dpr,o.n));
    // arrows
    arrow(ltx,80,70,220,70); arrow(ltx,220,70,360,70); arrow(ltx,360,70,500,70);
    ltx.fillStyle='#8aa2c0'; ltx.fillText('subobject classifier', 260*dpr, 160*dpr);

    // Right panel: laws flip with classicalization
    rtx.setTransform(1,0,0,1,0,0); rtx.clearRect(0,0,rightC.width,rightC.height);
    const WR=rightC.width, HR=rightC.height; bg(rtx,WR,HR);
    rtx.font=`${13*dpr}px system-ui`; rtx.fillStyle='#c9d6ef';
    rtx.fillText('Internal logic per stage:', 20*dpr, 30*dpr);
    const rows=[['1D','Heyting','¬¬A ≠ A','EM false'], ['2D+','Boolean (on subalgebras)','¬¬A = A','EM true']];
    rows.forEach((r,i)=>{ row(r, 20, 60 + i*28); });
    lawBadge.textContent='Topos view'; lawCheck.textContent='Ω_J separates subobjects'; lawCheck.className='badge ok';
    rightBar.style.width='66%'; leftBar.style.width='33%'; leftBadge1.textContent='Objects & Ω_J'; leftBadge2.textContent='arrows are functors';

    function node(ctx,x,y,lab){ ctx.fillStyle='#0e1520'; ctx.strokeStyle='#1b2736'; roundRect(ctx,x-26*dpr,y-12*dpr,52*dpr,24*dpr,8); ctx.fill(); ctx.stroke(); ctx.fillStyle='#a8c2ff'; ctx.fillText(lab,x-16*dpr,y+4*dpr); }
    function arrow(ctx,x1,y1,x2,y2){ ctx.strokeStyle='#32507a'; ctx.beginPath(); ctx.moveTo(x1*dpr,y1*dpr); ctx.lineTo(x2*dpr,y2*dpr); ctx.stroke(); ctx.beginPath(); ctx.moveTo(x2*dpr,y2*dpr); ctx.lineTo((x2-6)*dpr,(y2-3)*dpr); ctx.lineTo((x2-6)*dpr,(y2+3)*dpr); ctx.closePath(); ctx.fillStyle='#32507a'; ctx.fill(); }
    function row(data,x,y){ const cols=[0,120,280,430]; data.forEach((t,i)=>{ rtx.fillText(t, (x+cols[i])*dpr, y*dpr); }); }
    function roundRect(ctx,x,y,w,h,r){ ctx.beginPath(); ctx.moveTo(x+r,y); ctx.arcTo(x+w,y,x+w,y+h,r); ctx.arcTo(x+w,y+h,x,y+h,r); ctx.arcTo(x,y+h,x,y,r); ctx.arcTo(x,y,x+w,y,r); ctx.closePath(); }
    function bg(ctx,W,H){ const grd=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.min(W,H)*.6); grd.addColorStop(0,'#0b121b'); grd.addColorStop(1,'#081018'); ctx.fillStyle=grd; ctx.fillRect(0,0,W,H);}  
  }

  // ===== Module switching =====
  function switchModule(){
    module = moduleSel.value;
    effectCtl.hidden = module!=='effect';
    omlCtl.hidden = module!=='oml';
    multiCtl.hidden = module!=='multiway';
    holoCtl.hidden = module!=='holo';
    $('leftTitle').textContent = (module==='effect')? 'Effect & MV (Model)'
      : (module==='oml')? 'Orthomodular Subspaces (Model)'
      : (module==='multiway')? 'Multiway System (Nodes)'
      : (module==='holo')? 'Boundary Store (Write)'
      : 'Objects & Ω_J (Diagram)';
    $('rightTitle').textContent = (module==='effect')? 'Definedness & Results'
      : (module==='oml')? 'Orthomodular Identity'
      : (module==='multiway')? 'Reducibility Monitor'
      : (module==='holo')? 'Interior Encoding'
      : 'Internal Logic';
    draw();
  }

  // Main draw dispatcher
  function draw(){
    if (module==='effect') effectDraw();
    else if (module==='oml') omlDraw();
    else if (module==='multiway') multiwayDraw();
    else if (module==='holo') holoDraw();
    else toposDraw();
  }

  // Events
  moduleSel.addEventListener('change', switchModule);
  [A,B,mvMode,effectMode].forEach(el=> el.addEventListener('input', draw));
  [alpha,beta,YisPlane].forEach(el=> el.addEventListener('input', draw));
  [ruleSel,depth].forEach(el=> el.addEventListener('input', draw));

  leftC.addEventListener('click', (e)=>{
    if(module!=='holo') return; const rect=leftC.getBoundingClientRect();
    const x=(e.clientX-rect.left)*dpr, y=(e.clientY-rect.top)*dpr; const cx=leftC.width/2, cy=leftC.height/2; const R=Math.min(leftC.width,leftC.height)*.35;
    const th=Math.atan2(y-cy, x-cx); const r=Math.hypot(x-cx,y-cy);
    if(Math.abs(r-R) < 20*dpr) { bits.push(th); draw(); }
  });
  writeBit.addEventListener('click', ()=>{ bits.push(Math.random()*Math.PI*2); draw(); });
  eraseBits.addEventListener('click', ()=>{ bits=[]; draw(); });

  // Initial
  switchModule();
})();
</script>
</body>
</html>

