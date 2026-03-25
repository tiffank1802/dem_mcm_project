/**
 * Glass Displacement Effect for Navigation Bar
 * Generates dynamic SVG displacement map and applies filter
 */

class GlassDisplacementNav {
    constructor() {
        this.config = {
            width: window.innerWidth,
            height: 80, // nav height
            radius: 0,
            border: 0.07,
            lightness: 50,
            blur: 11,
            scale: -180,
            alpha: 0.93,
            blend: 'difference',
            x: 'R',
            y: 'B',
        };

        this.navElement = document.querySelector('.nav-liquid');
        this.init();
    }

    init() {
        // Create SVG container
        this.createSVGFilter();

        // Build displacement map
        this.buildDisplacementMap();

        // Apply filter
        this.applyFilter();

        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }

    createSVGFilter() {
        // Create SVG element with filter
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('style', 'display: none;');
        svg.setAttribute('width', '0');
        svg.setAttribute('height', '0');

        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');

        const filter = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'filter'
        );
        filter.setAttribute('id', 'glass-displacement-filter');
        filter.setAttribute('color-interpolation-filters', 'sRGB');

        // feImage for displacement map
        const feImage = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feImage'
        );
        feImage.setAttribute('x', '0');
        feImage.setAttribute('y', '0');
        feImage.setAttribute('width', '100%');
        feImage.setAttribute('height', '100%');
        feImage.setAttribute('result', 'map');
        feImage.setAttribute('preserveAspectRatio', 'xMidYMid slice');

        // RED channel displacement
        const feDispRed = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feDisplacementMap'
        );
        feDispRed.setAttribute('in', 'SourceGraphic');
        feDispRed.setAttribute('in2', 'map');
        feDispRed.setAttribute('id', 'redchannel');
        feDispRed.setAttribute('xChannelSelector', this.config.x);
        feDispRed.setAttribute('yChannelSelector', this.config.y);
        feDispRed.setAttribute('scale', this.config.scale);
        feDispRed.setAttribute('result', 'dispRed');

        const feColorMatrixRed = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feColorMatrix'
        );
        feColorMatrixRed.setAttribute('in', 'dispRed');
        feColorMatrixRed.setAttribute('type', 'matrix');
        feColorMatrixRed.setAttribute(
            'values',
            '1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0'
        );
        feColorMatrixRed.setAttribute('result', 'red');

        // GREEN channel displacement
        const feDispGreen = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feDisplacementMap'
        );
        feDispGreen.setAttribute('in', 'SourceGraphic');
        feDispGreen.setAttribute('in2', 'map');
        feDispGreen.setAttribute('id', 'greenchannel');
        feDispGreen.setAttribute('xChannelSelector', this.config.x);
        feDispGreen.setAttribute('yChannelSelector', this.config.y);
        feDispGreen.setAttribute('scale', this.config.scale + 10);
        feDispGreen.setAttribute('result', 'dispGreen');

        const feColorMatrixGreen = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feColorMatrix'
        );
        feColorMatrixGreen.setAttribute('in', 'dispGreen');
        feColorMatrixGreen.setAttribute('type', 'matrix');
        feColorMatrixGreen.setAttribute(
            'values',
            '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0'
        );
        feColorMatrixGreen.setAttribute('result', 'green');

        // BLUE channel displacement
        const feDispBlue = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feDisplacementMap'
        );
        feDispBlue.setAttribute('in', 'SourceGraphic');
        feDispBlue.setAttribute('in2', 'map');
        feDispBlue.setAttribute('id', 'bluechannel');
        feDispBlue.setAttribute('xChannelSelector', this.config.x);
        feDispBlue.setAttribute('yChannelSelector', this.config.y);
        feDispBlue.setAttribute('scale', this.config.scale + 20);
        feDispBlue.setAttribute('result', 'dispBlue');

        const feColorMatrixBlue = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feColorMatrix'
        );
        feColorMatrixBlue.setAttribute('in', 'dispBlue');
        feColorMatrixBlue.setAttribute('type', 'matrix');
        feColorMatrixBlue.setAttribute(
            'values',
            '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0'
        );
        feColorMatrixBlue.setAttribute('result', 'blue');

        // Blend channels
        const feBlendRG = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feBlend'
        );
        feBlendRG.setAttribute('in', 'red');
        feBlendRG.setAttribute('in2', 'green');
        feBlendRG.setAttribute('mode', this.config.blend);
        feBlendRG.setAttribute('result', 'rg');

        const feBlendRGB = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feBlend'
        );
        feBlendRGB.setAttribute('in', 'rg');
        feBlendRGB.setAttribute('in2', 'blue');
        feBlendRGB.setAttribute('mode', this.config.blend);
        feBlendRGB.setAttribute('result', 'output');

        // Gaussian blur
        const feGaussianBlur = document.createElementNS(
            'http://www.w3.org/2000/svg',
            'feGaussianBlur'
        );
        feGaussianBlur.setAttribute('in', 'output');
        feGaussianBlur.setAttribute('stdDeviation', '0.7');

        // Append all elements
        filter.appendChild(feImage);
        filter.appendChild(feDispRed);
        filter.appendChild(feColorMatrixRed);
        filter.appendChild(feDispGreen);
        filter.appendChild(feColorMatrixGreen);
        filter.appendChild(feDispBlue);
        filter.appendChild(feColorMatrixBlue);
        filter.appendChild(feBlendRG);
        filter.appendChild(feBlendRGB);
        filter.appendChild(feGaussianBlur);

        defs.appendChild(filter);
        svg.appendChild(defs);

        // Store reference
        this.svg = svg;
        this.feImage = feImage;

        // Append to document
        document.body.appendChild(svg);
    }

    buildDisplacementMap() {
        const width = this.config.width;
        const height = this.config.height;
        const radius = this.config.radius;
        const border = Math.min(width, height) * (this.config.border * 0.5);

        // Create SVG for displacement map
        const mapSvg = `
            <svg viewBox="0 0 ${width} ${height}" xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
                <defs>
                    <linearGradient id="red-grad" x1="100%" y1="0%" x2="0%" y2="0%">
                        <stop offset="0%" stop-color="#000"/>
                        <stop offset="100%" stop-color="red"/>
                    </linearGradient>
                    <linearGradient id="blue-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#000"/>
                        <stop offset="100%" stop-color="blue"/>
                    </linearGradient>
                </defs>
                <rect x="0" y="0" width="${width}" height="${height}" fill="black"></rect>
                <rect x="0" y="0" width="${width}" height="${height}" rx="${radius}" fill="url(#red-grad)" />
                <rect x="0" y="0" width="${width}" height="${height}" rx="${radius}" fill="url(#blue-grad)" style="mix-blend-mode: ${this.config.blend}" />
                <rect x="${border}" y="${border}" width="${width - border * 2}" height="${height - border * 2}" rx="${radius}" fill="hsl(0 0% ${this.config.lightness}% / ${this.config.alpha})" style="filter:blur(${this.config.blur}px)" />
            </svg>
        `;

        // Encode as data URI
        const encoded = encodeURIComponent(mapSvg);
        const dataUri = `data:image/svg+xml,${encoded}`;

        // Update feImage href
        this.feImage.setAttribute('href', dataUri);
        this.feImage.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', dataUri);
    }

    applyFilter() {
        // Filter is automatically applied via CSS backdrop-filter URL
        // The SVG filter is now available in the DOM
    }

    handleResize() {
        // Update width on resize
        this.config.width = window.innerWidth;
        this.buildDisplacementMap();
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new GlassDisplacementNav();
    });
} else {
    new GlassDisplacementNav();
}
