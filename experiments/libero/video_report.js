// Recorders retain every second physical frame at 10 fps (20 simulator steps/s).
for (const card of document.querySelectorAll('.episode')) {
  const video = card.querySelector('video');
  const cursor = card.querySelector('.playhead');
  if (!video || !cursor) continue;
  const series = JSON.parse(card.dataset.series || '[]');
  const length = Number(card.dataset.length);
  const label = card.querySelector('.live-values');
  let animation;
  function update() {
    const step = video.ended ? length : Math.min(length, video.currentTime * 20);
    const x = 30 + 395 * step / Math.max(1, length);
    cursor.setAttribute('x1', x.toFixed(2));
    cursor.setAttribute('x2', x.toFixed(2));
    // Show the last measured prefix rather than a prediction from a future frame.
    let point;
    for (const item of series) {
      if (item.step <= step) point = item;
    }
    label.textContent = `Playback step ${Math.floor(step)} / ${length}`;
    if (point) label.textContent += ` · Scored step ${point.step} · Progress ${(point.progress * 100).toFixed(1)}% · Predicted success ${(point.success_probability * 100).toFixed(1)}% · Ground truth reward ${point.ground_truth_reward}`;
  }
  function animate() {
    update();
    if (!video.paused && !video.ended) animation = requestAnimationFrame(animate);
  }
  video.addEventListener('play', () => {
    cancelAnimationFrame(animation);
    animate();
  });
  for (const event of ['loadedmetadata', 'timeupdate', 'seeking', 'seeked', 'pause', 'ended', 'emptied']) {
    video.addEventListener(event, update);
  }
  update();
}
