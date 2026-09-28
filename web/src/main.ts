import './styles.css';
import { App } from './ui/app';

const app = new App();
void app.start().catch(e => {
  console.error(e);
  const t = document.getElementById('toast');
  if (t) { t.textContent = 'Stillpoint failed to start: ' + ((e as Error).message ?? e); t.hidden = false; }
});
