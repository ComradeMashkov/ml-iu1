// Interactive slides for L3 and S7: ReLU sliders and the network X-ray.
// The X-ray runs the real S7 network (weights in generated/oct23/s7-net.js).

document.querySelectorAll(".relu-lab").forEach((lab) => {
  const w = lab.querySelector("[name=w]");
  const b = lab.querySelector("[name=b]");
  const line = lab.querySelector("polyline");
  const kink = lab.querySelector("circle");
  const readout = lab.querySelector(".relu-readout");
  // Plot area: x from -4 to 4, h from 0 to 4 mapped onto 400 x 240 pixels.
  const px = (x) => 40 + (x + 4) * 45;
  const py = (h) => 250 - Math.min(h, 4.4) * 50;

  function draw() {
    const wv = Number(w.value);
    const bv = Number(b.value);
    const points = [];
    for (let x = -4; x <= 4.001; x += 0.05) points.push(`${px(x)},${py(Math.max(0, wv * x + bv))}`);
    line.setAttribute("points", points.join(" "));
    const fmt = (v) => String(v).replace("-", "−");
    let text = `w = ${fmt(wv)}, b = ${fmt(bv)}. `;
    if (wv === 0) {
      kink.setAttribute("visibility", "hidden");
      text += `Сумма a = ${fmt(bv)} не зависит от x: излома нет.`;
    } else {
      const x0 = -bv / wv;
      const visible = Math.abs(x0) <= 4;
      kink.setAttribute("visibility", visible ? "visible" : "hidden");
      kink.setAttribute("cx", px(x0));
      kink.setAttribute("cy", py(0));
      text += `Излом там, где w·x + b = 0: x = ${fmt(-bv)} / ${fmt(wv)} = ${fmt(Math.round(x0 * 100) / 100)}.`;
      if (!visible) text += " Он за краем графика.";
    }
    readout.textContent = text;
  }
  [w, b].forEach((input) => input.addEventListener("input", draw));
  draw();
});

document.querySelectorAll(".xray").forEach((xray) => {
  const net = window.S7_NET;
  if (!net) return;
  const pad = xray.querySelector(".xray-pad");
  const padContext = pad.getContext("2d");
  const input = xray.querySelector(".xray-input");
  const tiles = [...xray.querySelectorAll(".xray-tile")];
  const verdict = xray.querySelector(".xray-verdict");
  const scoreLine = xray.querySelector(".xray-score");
  let strokes = [];
  let drawing = false;

  // Each tile shows the 784 input weights of one hidden neuron:
  // red where ink raises its sum, blue where ink lowers it.
  tiles.forEach((tile, i) => {
    const canvas = tile.querySelector("canvas");
    const context = canvas.getContext("2d");
    const image = context.createImageData(28, 28);
    for (let j = 0; j < 784; j++) {
      const v = Math.max(-1, Math.min(1, net.W1[i * 784 + j] / 0.15));
      const k = j * 4;
      image.data[k] = v > 0 ? 255 : Math.round(255 * (1 + v));
      image.data[k + 1] = Math.round(255 * (1 - Math.abs(v)));
      image.data[k + 2] = v < 0 ? 255 : Math.round(255 * (1 - v));
      image.data[k + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  });

  function run(x) {
    const inputContext = input.getContext("2d");
    const picture = inputContext.createImageData(28, 28);
    x.forEach((v, j) => {
      const k = j * 4;
      const shade = Math.round(255 * (1 - v));
      picture.data[k] = picture.data[k + 1] = picture.data[k + 2] = shade;
      picture.data[k + 3] = 255;
    });
    inputContext.putImageData(picture, 0, 0);
    if (!x.some((v) => v > 0)) {
      tiles.forEach((tile) => {
        tile.style.opacity = 0.25;
        tile.querySelector("span").textContent = "0";
      });
      verdict.textContent = "Нарисуйте или выберите рисунок";
      scoreLine.textContent = "";
      return;
    }
    let z = net.b2;
    const votes = tiles.map((tile, i) => {
      let a = net.b1[i];
      for (let j = 0; j < 784; j++) a += net.W1[i * 784 + j] * x[j];
      const h = Math.max(0, a);
      const vote = net.w2[i] * h;
      z += vote;
      return { tile, h, vote };
    });
    const strongest = Math.max(...votes.map((v) => v.h), 1e-9);
    votes.forEach(({ tile, h, vote }) => {
      tile.style.opacity = 0.35 + 0.65 * (h / strongest);
      const label = tile.querySelector("span");
      label.textContent = h === 0 ? "молчит" : (vote >= 0 ? "+" : "−") + Math.abs(vote).toFixed(1);
      label.className = h === 0 ? "" : vote >= 0 ? "teapot" : "cat";
    });
    const p = 1 / (1 + Math.exp(-z));
    verdict.textContent = p >= 0.5 ? `Чайник, p = ${p.toFixed(3)}` : `Кот, 1 − p = ${(1 - p).toFixed(3)}`;
    scoreLine.textContent = `z = ${net.b2.toFixed(2)} + сумма голосов = ${z.toFixed(2)}`;
  }

  // Bring a free-hand drawing to the QuickDraw bitmap format:
  // fit the longer side into about 24.5 of 28 pixels, centre it, blur to 28 x 28.
  const big = document.createElement("canvas");
  big.width = big.height = 280;
  const bigContext = big.getContext("2d", { willReadFrequently: true });

  function strokesToInput() {
    const points = strokes.flat();
    if (points.length === 0) return new Array(784).fill(0);
    const xs = points.map((p) => p[0]);
    const ys = points.map((p) => p[1]);
    const left = Math.min(...xs);
    const top = Math.min(...ys);
    const side = Math.max(Math.max(...xs) - left, Math.max(...ys) - top, 1);
    const scale = 245 / side;
    const dx = 140 - ((Math.max(...xs) - left) * scale) / 2;
    const dy = 140 - ((Math.max(...ys) - top) * scale) / 2;
    bigContext.clearRect(0, 0, 280, 280);
    bigContext.lineWidth = 16;
    bigContext.lineCap = bigContext.lineJoin = "round";
    bigContext.strokeStyle = "black";
    strokes.forEach((stroke) => {
      bigContext.beginPath();
      stroke.forEach(([sx, sy], i) => {
        const tx = dx + (sx - left) * scale;
        const ty = dy + (sy - top) * scale;
        if (i === 0) bigContext.moveTo(tx, ty + 0.01);
        bigContext.lineTo(tx, ty);
      });
      bigContext.stroke();
    });
    const alpha = bigContext.getImageData(0, 0, 280, 280).data;
    const x = new Array(784).fill(0);
    for (let r = 0; r < 280; r++)
      for (let c = 0; c < 280; c++) x[Math.floor(r / 10) * 28 + Math.floor(c / 10)] += alpha[(r * 280 + c) * 4 + 3];
    const brightest = Math.max(...x);
    return x.map((v) => (brightest > 0 ? v / brightest : 0));
  }

  function paintStrokes() {
    padContext.clearRect(0, 0, pad.width, pad.height);
    padContext.lineWidth = 9;
    padContext.lineCap = padContext.lineJoin = "round";
    strokes.forEach((stroke) => {
      padContext.beginPath();
      stroke.forEach(([sx, sy], i) => {
        if (i === 0) padContext.moveTo(sx, sy + 0.01);
        padContext.lineTo(sx, sy);
      });
      padContext.stroke();
    });
  }

  function position(event) {
    const box = pad.getBoundingClientRect();
    return [((event.clientX - box.left) * pad.width) / box.width, ((event.clientY - box.top) * pad.height) / box.height];
  }

  let pending = false;
  function update() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      paintStrokes();
      run(strokesToInput());
    });
  }

  pad.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    event.stopPropagation();
    pad.setPointerCapture(event.pointerId);
    drawing = true;
    strokes.push([position(event)]);
    update();
  });
  pad.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    event.stopPropagation();
    strokes[strokes.length - 1].push(position(event));
    update();
  });
  ["pointerup", "pointercancel"].forEach((name) => pad.addEventListener(name, () => { drawing = false; }));

  xray.querySelector("[data-clear]").addEventListener("click", () => {
    strokes = [];
    paintStrokes();
    run(new Array(784).fill(0));
  });

  xray.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      const sample = net.samples[Number(button.dataset.sample)];
      strokes = [];
      padContext.clearRect(0, 0, pad.width, pad.height);
      const cell = pad.width / 28;
      sample.pixels.forEach((v, j) => {
        padContext.fillStyle = `rgba(0,0,0,${v / 255})`;
        padContext.fillRect((j % 28) * cell, Math.floor(j / 28) * cell, cell, cell);
      });
      run(sample.pixels.map((v) => v / 255));
    });
  });

  run(new Array(784).fill(0));
});
