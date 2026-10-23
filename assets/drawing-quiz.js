document.querySelectorAll(".drawing-quiz").forEach((quiz) => {
  const drawings = [...quiz.querySelectorAll("[data-drawing]")];
  const choices = [...quiz.querySelectorAll("[data-guess]")];
  const progress = quiz.querySelector(".drawing-quiz-progress");
  const feedback = quiz.querySelector('[role="status"]');
  const next = quiz.querySelector("[data-next]");
  let index = 0;

  function showDrawing() {
    drawings.forEach((drawing, i) => { drawing.hidden = i !== index; });
    progress.textContent = `Рисунок ${index + 1} из ${drawings.length}`;
    feedback.textContent = "Выберите вариант, чтобы открыть метку набора.";
    choices.forEach((choice) => {
      choice.disabled = false;
      choice.setAttribute("aria-pressed", "false");
    });
    next.hidden = true;
    next.textContent = index === drawings.length - 1 ? "Пройти снова" : "Следующий рисунок";
  }

  choices.forEach((choice) => {
    choice.addEventListener("click", () => {
      const answer = drawings[index].dataset.answer;
      const guess = choice.dataset.guess;
      feedback.textContent = guess === answer
        ? `Совпало. Метка набора: «${answer}».`
        : `Метка набора: «${answer}». Ваш вариант: «${guess}».`;
      if (drawings[index].dataset.net) feedback.textContent += ` Сеть ответила: ${drawings[index].dataset.net}.`;
      choices.forEach((button) => {
        button.disabled = true;
        button.setAttribute("aria-pressed", String(button === choice));
      });
      next.hidden = false;
      next.focus({ preventScroll: true });
    });
  });

  next.addEventListener("click", () => {
    index = (index + 1) % drawings.length;
    showDrawing();
    choices[0].focus({ preventScroll: true });
  });

  showDrawing();
});
