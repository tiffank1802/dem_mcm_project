/**
 * Liquid Glass Card Engine
 * Génère dynamiquement les displacement maps SVG
 * et les applique aux cartes via backdrop-filter
 */

class LiquidGlassEngine {
    constructor() {
        this.cards = document.querySelectorAll('.liquid-glass-card');
        this.config = {
            scale: -120,
            border: 0.08,
            lightness: 50,
            alpha: 0.9,
            blur: 10,
            blend: 'difference',
            // Aberration chromatique
            redOffset: 0,
            greenOffset: 10,
            blueOffset: 20,
        };

        this.init();
    }

    /**
     * Construit l'image SVG de displacement pour une carte
     */
    buildDisplacementSVG(width, height, radius) {
        const border = Math.min(width, height) * (this.config.border * 0.5);

        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">
            <defs>
                <linearGradient id="lg-red" x1="100%" y1="0%" x2="0%" y2="0%">
                    <stop offset="0%" stop-color="#000"/>
                    <stop offset="100%" stop-color="red"/>
                </linearGradient>
                <linearGradient id="lg-blue" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#000"/>
                    <stop offset="100%" stop-color="blue"/>
                </linearGradient>
            </defs>
            <rect x="0" y="0" width="${width}" height="${height}" fill="black"/>
            <rect x="0" y="0" width="${width}" height="${height}"
                  rx="${radius}" fill="url(#lg-red)"/>
            <rect x="0" y="0" width="${width}" height="${height}"
                  rx="${radius}" fill="url(#lg-blue)"
                  style="mix-blend-mode: ${this.config.blend}"/>
            <rect x="${border}" y="${border}"
                  width="${width - border * 2}" height="${height - border * 2}"
                  rx="${radius}"
                  fill="hsl(0 0% ${this.config.lightness}% / ${this.config.alpha})"
                  style="filter:blur(${this.config.blur}px)"/>
        </svg>`;
    }

    /**
     * Encode le SVG en data URI
     */
    svgToDataURI(svgString) {
        return `data:image/svg+xml,${encodeURIComponent(svgString)}`;
    }

    /**
     * Obtient les dimensions réelles d'une carte
     */
    getCardDimensions(card) {
        const rect = card.getBoundingClientRect();
        const style = getComputedStyle(card);
        const radius = parseFloat(style.borderRadius) || 15;

        return {
            width: Math.round(rect.width),
            height: Math.round(rect.height),
            radius: radius,
        };
    }

    /**
     * Applique le displacement à un filtre SVG
     */
    applyDisplacementToFilter(filterSelector, dims) {
        const svg = this.buildDisplacementSVG(dims.width, dims.height, dims.radius);
        const dataUri = this.svgToDataURI(svg);

        // Mettre à jour les feImage
        const feImages = document.querySelectorAll(`${filterSelector} feImage`);
        feImages.forEach(feImage => {
            feImage.setAttribute('href', dataUri);
        });

        // Mettre à jour les scales des displacement maps
        const redChannel = document.querySelector(`${filterSelector} [id$="-red"]`);
        const greenChannel = document.querySelector(`${filterSelector} [id$="-green"]`);
        const blueChannel = document.querySelector(`${filterSelector} [id$="-blue"]`);

        if (redChannel) {
            redChannel.setAttribute('scale', this.config.scale + this.config.redOffset);
        }
        if (greenChannel) {
            greenChannel.setAttribute('scale', this.config.scale + this.config.greenOffset);
        }
        if (blueChannel) {
            blueChannel.setAttribute('scale', this.config.scale + this.config.blueOffset);
        }
    }

    /**
     * Affiche la preview du displacement map
     */
    showDisplacementPreview(card, dims) {
        const previewEl = document.getElementById('card-displacement-preview');
        if (!previewEl) return;

        const svg = this.buildDisplacementSVG(dims.width, dims.height, dims.radius);
        previewEl.innerHTML = svg;
    }

    /**
     * Initialise tout
     */
    init() {
        // Attendre que les cartes soient rendues
        requestAnimationFrame(() => {
            this.updateAllCards();
        });

        // Recalculer au resize
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => this.updateAllCards(), 200);
        });

        // Observer les changements de taille
        if (typeof ResizeObserver !== 'undefined') {
            const observer = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    if (entry.target.classList.contains('liquid-glass-card')) {
                        this.updateCard(entry.target);
                    }
                }
            });

            this.cards.forEach(card => observer.observe(card));
        }

        console.log(`🔮 Liquid Glass Engine: ${this.cards.length} cards initialized`);
    }

    /**
     * Met à jour une carte spécifique
     */
    updateCard(card) {
        const dims = this.getCardDimensions(card);

        // Pour la première carte, montrer la preview
        if (card.dataset.cardId === 'social') {
            this.showDisplacementPreview(card, dims);
        }

        // Appliquer au filtre global
        this.applyDisplacementToFilter('#liquid-glass', dims);
        this.applyDisplacementToFilter('#icon-glass', {
            width: 60,
            height: 60,
            radius: 30,
        });
    }

    /**
     * Met à jour toutes les cartes
     */
    updateAllCards() {
        // Utiliser les dimensions de la première carte pour le filtre global
        const firstCard = this.cards[0];
        if (firstCard) {
            this.updateCard(firstCard);
        }
    }
}

// ============================================
// INITIALISATION
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Lancer le moteur liquid glass
    const engine = new LiquidGlassEngine();

    // Ajouter des effets d'interaction supplémentaires
    addMouseTrackingEffects();
    addParallaxOnScroll();
});

/**
 * Effet de suivi de souris sur les cartes
 * Crée un reflet dynamique qui suit le curseur
 */
function addMouseTrackingEffects() {
    const cards = document.querySelectorAll('.liquid-glass-card');

    cards.forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = ((e.clientX - rect.left) / rect.width) * 100;
            const y = ((e.clientY - rect.top) / rect.height) * 100;

            // Reflet dynamique qui suit la souris
            card.style.setProperty('--mouse-x', `${x}%`);
            card.style.setProperty('--mouse-y', `${y}%`);

            // Rotation 3D subtile
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const deltaX = (e.clientX - rect.left - centerX) / centerX;
            const deltaY = (e.clientY - rect.top - centerY) / centerY;

            card.style.transform = `
                perspective(800px)
                rotateY(${deltaX * 5}deg)
                rotateX(${-deltaY * 5}deg)
                translateY(-2px)
            `;
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
            card.style.removeProperty('--mouse-x');
            card.style.removeProperty('--mouse-y');
        });
    });
}

/**
 * Effet parallax subtil au scroll
 */
function addParallaxOnScroll() {
    const cards = document.querySelectorAll('.liquid-glass-card');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('glass-visible');
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px',
    });

    cards.forEach((card, index) => {
        card.style.setProperty('--appear-delay', `${index * 0.1}s`);
        observer.observe(card);
    });
}
