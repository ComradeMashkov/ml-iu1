(() => {
  let framePending = false;

  const scheduleLayout = () => {
    if (framePending) return;
    framePending = true;
    requestAnimationFrame(() => {
      framePending = false;
      if (typeof window.Reveal?.layout === "function") {
        window.Reveal.layout();
      }
    });
  };

  // Lazy images can finish loading after Reveal has centered the slide.
  document.addEventListener("load", (event) => {
    if (event.target instanceof HTMLImageElement && event.target.closest(".reveal")) {
      scheduleLayout();
    }
  }, true);

  window.addEventListener("load", scheduleLayout);
  if (document.fonts) {
    document.fonts.ready.then(scheduleLayout);
    document.fonts.addEventListener("loadingdone", scheduleLayout);
  }
})();
