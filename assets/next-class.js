(() => {
  "use strict";

  const target = document.getElementById("next-class-content");
  const schedule = document.querySelector(".schedule-calendar table");
  if (!target || !schedule) return;

  const parts = new Intl.DateTimeFormat("ru-RU", {
    timeZone: "Europe/Moscow",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date());
  const dateParts = Object.fromEntries(parts.map(({ type, value }) => [type, value]));
  const today = `${dateParts.year}-${dateParts.month}-${dateParts.day}`;

  const rows = [...schedule.querySelectorAll("tbody tr")]
    .map((row) => ({
      row,
      date: row.querySelector("time[datetime]")?.getAttribute("datetime"),
    }))
    .filter(({ date }) => date);
  const next = rows.find(({ date }) => date >= today);

  if (!next) {
    const message = document.createElement("p");
    message.className = "muted";
    message.textContent = "Все запланированные учебные дни завершены.";
    target.replaceChildren(message);
    return;
  }

  const cells = [...next.row.cells];
  const dateLine = document.createElement("p");
  dateLine.className = "next-class-date";

  const dateTitle = document.createElement("strong");
  dateTitle.textContent = next.date === today ? `Сегодня · ${cells[0].textContent.trim()}` : cells[0].textContent.trim();
  const week = document.createElement("span");
  week.textContent = `Пятница · учебная неделя ${cells[1].textContent.trim()}`;
  dateLine.append(dateTitle, week);

  const times = ["15:55–17:25", "17:35–19:05", "19:15–20:45"];
  const slots = document.createElement("div");
  slots.className = "next-class-slots";

  cells.slice(2, 5).forEach((cell, index) => {
    const slot = document.createElement("div");
    slot.className = "next-class-slot";
    const time = document.createElement("span");
    time.className = "next-class-time";
    time.textContent = times[index];
    const topic = document.createElement("div");
    topic.className = "next-class-topic";
    [...cell.childNodes].forEach((node) => topic.append(node.cloneNode(true)));
    slot.append(time, topic);
    slots.append(slot);
  });

  target.replaceChildren(dateLine, slots);
})();
