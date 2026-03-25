/* ═══════════════════════════════════════════════════════
   Liquid Glass — SVG Displacement Filter
   Applies glass distortion effect to elements with
   .glass-displace class via SVG backdrop-filter
   ═══════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // Config
  const CONFIG = {
    width: 336,
    height: 96,
    radius: 16,
    border: 0.07,
    lightness: 50,
    alpha: 0.93,
    blur: 11,
    scale: -180,
    blend: 'difference',
    x: 'R',
    y: 'B',
    r: 0,
    g: 10,
    b: 20,
  };

  // Build SVG displacement map as data URI
  function buildDisplacementURI() {
    const w = CONFIG.width;
    const h = CONFIG.height;
    const r = CONFIG.radius;
    const border = Math.min(w, h) * (CONFIG.border * 0.5);

    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">
      <defs>
        <linearGradient id="r" x1="100%" y1="0%" x2="0%" y2="0%">
          <stop offset="0%" stop-color="#000"/>
          <stop offset="100%" stop-color="red"/>
        </linearGradient>
        <linearGradient id="b" x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stop-color="#000"/>
          <stop offset="100%" stop-color="blue"/>
        </linearGradient>
      </defs>
      <rect width="${w}" height="${h}" fill="black"/>
      <rect width="${w}" height="${h}" rx="${r}" fill="url(#r)"/>
      <rect width="${w}" height="${h}" rx="${r}" fill="url(#b)" style="mix-blend-mode:${CONFIG.blend}"/>
      <rect x="${border}" y="${border}" width="${w - border * 2}" height="${h - border * 2}" rx="${r}"
            fill="hsl(0 0% ${CONFIG.lightness}% / ${CONFIG.alpha})"
            style="filter:blur(${CONFIG.blur}px)"/>
    </svg>`;

    return 'data:image/svg+xml,' + encodeURIComponent(svg);
  }

  // Inject SVG filter into DOM
  function injectFilter() {
    if (document.getElementById('liquid-glass-filter')) return;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '0');
    svg.setAttribute('height', '0');
    svg.style.position = 'absolute';
    svg.innerHTML = `
      <defs>
        <filter id="liquid-glass-filter" color-interpolation-filters="sRGB">
          <feImage x="0" y="0" width="100%" height="100%" result="map" id="liquid-glass-fe-image"/>
          <feDisplacementMap in="SourceGraphic" in2="map" id="liquid-glass-red"
            xChannelSelector="R" yChannelSelector="G" result="dispRed"/>
          <feColorMatrix in="dispRed" type="matrix"
            values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" result="red"/>
          <feDisplacementMap in="SourceGraphic" in2="map" id="liquid-glass-green"
            xChannelSelector="R" yChannelSelector="G" result="dispGreen"/>
          <feColorMatrix in="dispGreen" type="matrix"
            values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0" result="green"/>
          <feDisplacementMap in="SourceGraphic" in2="map" id="liquid-glass-blue"
            xChannelSelector="R" yChannelSelector="G" result="dispBlue"/>
          <feColorMatrix in="dispBlue" type="matrix"
            values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0" result="blue"/>
          <feBlend in="red" in2="green" mode="screen" result="rg"/>
          <feBlend in="rg" in2="blue" mode="screen" result="output"/>
          <feGaussianBlur in="output" stdDeviation="0.7"/>
        </filter>
      </defs>`;

    document.body.appendChild(svg);
  }

  // Update displacement map
  function updateDisplacement() {
    const uri = buildDisplacementURI();
    const feImage = document.getElementById('liquid-glass-fe-image');
    if (feImage) feImage.setAttribute('href', uri);

    const red = document.getElementById('liquid-glass-red');
    const green = document.getElementById('liquid-glass-green');
    const blue = document.getElementById('liquid-glass-blue');

    if (red) red.setAttribute('scale', CONFIG.scale + CONFIG.r);
    if (green) green.setAttribute('scale', CONFIG.scale + CONFIG.g);
    if (blue) blue.setAttribute('scale', CONFIG.scale + CONFIG.b);
  }

  // Theme toggle
  function initTheme() {
    const stored = localStorage.getItem('theme');
    if (stored) {
      document.documentElement.dataset.theme = stored;
    } else if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.dataset.theme = 'dark';
    }
  }

  window.toggleTheme = function () {
    const current = document.documentElement.dataset.theme;
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('theme', next);
  };

  // Init
  function init() {
    injectFilter();
    updateDisplacement();
    initTheme();

    // Observe resize for displacement map
    const ro = new ResizeObserver(() => {
      const el = document.querySelector('.glass-displace');
      if (el) {
        CONFIG.width = el.offsetWidth || 336;
        CONFIG.height = el.offsetHeight || 96;
        updateDisplacement();
      }
    });

    document.querySelectorAll('.glass-displace').forEach((el) => {
      ro.observe(el);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
