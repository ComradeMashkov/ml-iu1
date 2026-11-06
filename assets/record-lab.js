// Interactive slides for L4: record history of Super Mario Bros. Any% (data in generated/nov06/smb.js).

const SVG = "http://www.w3.org/2000/svg";
const minus = (v) => String(v).replace("-", "−");

// One plot: years 2001.5-2027.5 across, seconds [lo, hi] up, in a 640 x 340 viewBox.
function plot(svg, lo, hi, ticks) {
  const px = (t) => 56 + ((t - 2001.5) / 26) * 570;
  const py = (s) => 300 - ((s - lo) / (hi - lo)) * 280;
  const add = (name, attrs, parent = svg) => {
    const node = document.createElementNS(SVG, name);
    Object.entries(attrs).forEach(([k, v]) => node.setAttribute(k, v));
    parent.appendChild(node);
    return node;
  };
  const clipId = `clip-${Math.random().toString(36).slice(2)}`;
  add("rect", { x: 56, y: 20, width: 570, height: 280 }, add("clipPath", { id: clipId }, add("defs", {})));
  add("line", { x1: 56, y1: 300, x2: 626, y2: 300, class: "axis" });
  add("line", { x1: 56, y1: 20, x2: 56, y2: 300, class: "axis" });
  [2005, 2010, 2015, 2020, 2025].forEach((t) => {
    add("text", { x: px(t) - 16, y: 322 }).textContent = t;
  });
  ticks.forEach((s) => {
    add("line", { x1: 56, y1: py(s), x2: 626, y2: py(s), class: "grid" });
    add("text", { x: 4, y: py(s) + 5 }).textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  });
  const layer = add("g", { "clip-path": `url(#${clipId})` });
  return { px, py, add, layer };
}

function dots(p, color = (i) => "rec") {
  return window.SMB.year.map((t, i) => p.add("circle", { cx: p.px(t), cy: p.py(window.SMB.sec[i]), r: 5, class: color(i) }, p.layer));
}

function fit(indices) {
  const { year, sec } = window.SMB;
  const n = indices.length;
  const mx = indices.reduce((a, i) => a + year[i], 0) / n;
  const my = indices.reduce((a, i) => a + sec[i], 0) / n;
  let sxy = 0;
  let sxx = 0;
  indices.forEach((i) => {
    sxy += (year[i] - mx) * (sec[i] - my);
    sxx += (year[i] - mx) ** 2;
  });
  const slope = sxy / sxx;
  return { slope, at: (t) => my + slope * (t - mx) };
}

document.querySelectorAll(".line-lab").forEach((lab) => {
  if (!window.SMB) return;
  const p = plot(lab.querySelector("svg"), 270, 330, [270, 280, 290, 300, 310, 320, 330]);
  dots(p);
  const line = p.add("line", { class: "fit" }, p.layer);
  const level = lab.querySelector("[name=level]");
  const slope = lab.querySelector("[name=slope]");
  const readout = lab.querySelector(".lab-readout");
  function draw() {
    const v = Number(level.value);
    const b = Number(slope.value);
    const at = (t) => v + b * (t - 2014);
    line.setAttribute("x1", p.px(2001.5));
    line.setAttribute("y1", p.py(at(2001.5)));
    line.setAttribute("x2", p.px(2027.5));
    line.setAttribute("y2", p.py(at(2027.5)));
    const { year, sec } = window.SMB;
    const mse = year.reduce((a, t, i) => a + (at(t) - sec[i]) ** 2, 0) / year.length;
    let text = `время = ${v.toFixed(1)} ${b < 0 ? "−" : "+"} ${Math.abs(b).toFixed(2)}·(год − 2014). `;
    text += `Средний квадрат ошибки по ${year.length} рекордам: ${mse.toFixed(1)} с². `;
    text += b < 0 ? `Ноль секунд в ${Math.floor(2014 + v / -b)} году.` : "Прямая не опускается: ноль не наступит.";
    readout.textContent = text;
  }
  lab.querySelector("[data-best]").addEventListener("click", () => {
    level.value = lab.dataset.level;
    slope.value = lab.dataset.slope;
    draw();
  });
  [level, slope].forEach((input) => input.addEventListener("input", draw));
  draw();
});

document.querySelectorAll(".window-lab").forEach((lab) => {
  if (!window.SMB) return;
  const p = plot(lab.querySelector("svg"), 290, 330, [290, 300, 310, 320, 330]);
  const from = lab.querySelector("[name=from]");
  const to = lab.querySelector("[name=to]");
  const marks = dots(p);
  const line = p.add("line", { class: "fit" }, p.layer);
  const readout = lab.querySelector(".lab-readout");
  function draw() {
    const a = Number(from.value);
    const b = Number(to.value);
    const chosen = [];
    window.SMB.year.forEach((t, i) => {
      const inside = t >= a && t < b + 1;
      marks[i].setAttribute("class", inside ? "rec" : "rec off");
      if (inside) chosen.push(i);
    });
    if (chosen.length < 3) {
      line.setAttribute("visibility", "hidden");
      readout.textContent = `С ${a} по ${b} год: ${chosen.length} рек. Для прямой возьмите хотя бы 3.`;
      return;
    }
    const { slope, at } = fit(chosen);
    line.setAttribute("visibility", "visible");
    line.setAttribute("x1", p.px(2001.5));
    line.setAttribute("y1", p.py(at(2001.5)));
    line.setAttribute("x2", p.px(2027.5));
    line.setAttribute("y2", p.py(at(2027.5)));
    const zero = slope < 0 ? Math.floor(2002 + (at(2002) / -slope)) : null;
    readout.textContent = `С ${a} по ${b} год: ${chosen.length} рек. Наклон ${minus(slope.toFixed(3))} с в год. `
      + (zero ? `Ноль секунд в ${zero} году.` : "Прямая не опускается.");
  }
  [from, to].forEach((input) => input.addEventListener("input", draw));
  draw();
});

document.querySelectorAll(".degree-lab").forEach((lab) => {
  if (!window.SMB) return;
  const p = plot(lab.querySelector("svg"), 285, 335, [290, 300, 310, 320, 330]);
  p.add("rect", { x: p.px(2020), y: 20, width: p.px(2027.5) - p.px(2020), height: 280, class: "future" }, p.layer);
  dots(p, (i) => (window.SMB.year[i] < 2020 ? "rec" : "rec later"));
  const curve = p.add("polyline", { class: "fit", fill: "none" }, p.layer);
  const degree = lab.querySelector("[name=degree]");
  const readout = lab.querySelector(".lab-readout");
  const label = lab.querySelector(".degree-value");
  function draw() {
    const d = Number(degree.value);
    const item = window.SMB.poly[d];
    // Values far outside the plot are clamped so the polyline still reaches the edge.
    curve.setAttribute("points", window.SMB.grid
      .map((t, j) => `${p.px(t)},${Math.max(-400, Math.min(800, p.py(item.curve[j])))}`).join(" "));
    label.textContent = d;
    const pred = item.pred_last;
    const shown = Math.abs(pred) < 10000 ? `${minus(pred.toFixed(1))} с` : `${minus(Math.round(pred).toLocaleString("ru-RU"))} с`;
    readout.innerHTML = `Ошибка на обучении (RMSE, 33 рекорда до 2020): <b>${item.train_rmse.toFixed(2)} с</b>.<br>`
      + `Прогноз на 14 августа 2026: <b>${shown}</b>. Факт: ${window.SMB.last.sec} с.`;
  }
  degree.addEventListener("input", draw);
  draw();
});

document.querySelectorAll(".task-quiz").forEach((quiz) => {
  const cards = [...quiz.querySelectorAll("[data-card]")];
  const choices = [...quiz.querySelectorAll("[data-guess]")];
  const progress = quiz.querySelector(".task-quiz-progress");
  const feedback = quiz.querySelector('[role="status"]');
  const next = quiz.querySelector("[data-next]");
  const rival = quiz.dataset.rivalName;
  let index = 0;
  let room = 0;
  let robot = 0;
  function count() {
    const done = rival ? ` · Зал ${room}, ${rival} ${robot}` : "";
    progress.textContent = `Вопрос ${index + 1} из ${cards.length}${done}`;
  }
  function show() {
    cards.forEach((card, i) => { card.hidden = i !== index; });
    count();
    feedback.textContent = "";
    choices.forEach((c) => { c.disabled = false; c.setAttribute("aria-pressed", "false"); });
    next.hidden = true;
    next.textContent = index === cards.length - 1 ? "Пройти снова" : "Следующий вопрос";
  }
  choices.forEach((choice) => choice.addEventListener("click", () => {
    const card = cards[index];
    const right = choice.dataset.guess === card.dataset.answer;
    feedback.textContent = `${right ? "Верно" : `Ответ: ${card.dataset.answer.toLowerCase()}`}. ${card.dataset.why}`;
    choices.forEach((c) => { c.disabled = true; c.setAttribute("aria-pressed", String(c === choice)); });
    if (right) room += 1;
    robot += Number(card.dataset.rival || 0);
    count();
    next.hidden = false;
    next.focus({ preventScroll: true });
  }));
  next.addEventListener("click", () => {
    index = (index + 1) % cards.length;
    if (index === 0) { room = 0; robot = 0; }
    show();
    choices[0].focus({ preventScroll: true });
  });
  show();
});
