document.querySelectorAll('[aria-disabled="true"]').forEach((link) => {
  link.addEventListener('click', (event) => event.preventDefault());
});

const motivationVideo = document.querySelector('[data-play-on-scroll]');
const motivationSection = motivationVideo?.closest('section');

if (motivationSection && 'IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    if (!entries.some((entry) => entry.isIntersecting)) return;
    observer.disconnect();
    motivationVideo.muted = true;
    // Keep the controls available if the browser blocks automatic playback.
    motivationVideo.play().catch(() => {});
  }, { threshold: 0.1 });

  // Manual playback also completes the one-time scroll trigger.
  motivationVideo.addEventListener('play', () => observer.disconnect(), { once: true });
  observer.observe(motivationSection);
}

document.querySelectorAll('[data-copy]').forEach((button) => {
  button.addEventListener('click', async () => {
    const target = document.getElementById(button.dataset.copy);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.innerText);
      const original = button.textContent;
      button.textContent = 'Copied';
      window.setTimeout(() => { button.textContent = original; }, 1600);
    } catch {
      button.textContent = 'Select text to copy';
    }
  });
});
