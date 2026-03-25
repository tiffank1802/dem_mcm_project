/**
 * Liquid Glass Card Engine
 * Génère dynamiquement les displacement maps SVG
 * et les applique aux cartes via backdrop-filter
 * 
 * Pour les cartes des méthodes de partitionnement dans le dashboard
 */

class LiquidGlassCardEngine {
    constructor(selector = '.method-card-liquid') {
        this.selector = selector;
        this.cards = document.querySelectorAll(selector);
        
        this.config = {
            // Paramètres de displacement
            scale: -120,
            border: 0.08,
            lightness: 50,
            alpha: 0.9,
            blur: 10,
            blend: 'difference',
            
            // Aberration chromatique (offset par canal)
            redOffset: 0,
            greenOffset: 10,
            blueOffset: 20,
            
            // Dimensions SVG
            filterWidth: 400,
            filterHeight: 300,
            borderRadius: 20,
        };

        this.init();
    }

    /**
     * Construit l'image SVG de displacement pour les cartes
     */
    buildDisplacementSVG(width, height, radius) {
        const border = Math.min(width, height) * (this.config.border * 0.5);
        const w = width || this.config.filterWidth;
        const h = height || this.config.filterHeight;
        const r = radius || this.config.borderRadius;

        return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w} ${h}">
            <defs>
                <!-- Gradient rouge (déplacement horizontal) -->
                <linearGradient id="lg-red" x1="100%" y1="0%" x2="0%" y2="0%">
                    <stop offset="0%" stop-color="#000"/>
                    <stop offset="50%" stop-color="#555"/>
                    <stop offset="100%" stop-color="red"/>
                </linearGradient>
                
                <!-- Gradient bleu (déplacement vertical) -->
                <linearGradient id="lg-blue" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#000"/>
                    <stop offset="50%" stop-color="#555"/>
                    <stop offset="100%" stop-color="blue"/>
                </linearGradient>
                
                <!-- Gradient radial pour effet de bulles liquides -->
                <radialGradient id="lg-radial" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="white" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="black" stop-opacity="0.2"/>
                </radialGradient>
            </defs>
            
            <!-- Fond noir (aucun déplacement) -->
            <rect x="0" y="0" width="${w}" height="${h}" fill="black"/>
            
            <!-- Forme arrondie avec gradient rouge -->
            <rect x="0" y="0" width="${w}" height="${h}"
                  rx="${r}" fill="url(#lg-red)"/>
            
            <!-- Superposition avec gradient bleu (blend mode) -->
            <rect x="0" y="0" width="${w}" height="${h}"
                  rx="${r}" fill="url(#lg-blue)"
                  style="mix-blend-mode: ${this.config.blend}"/>
            
            <!-- Zone stable au centre (pas de distorsion) -->
            <rect x="${border}" y="${border}"
                  width="${w - border * 2}" height="${h - border * 2}"
                  rx="${r}"
                  fill="hsl(0 0% ${this.config.lightness}% / ${this.config.alpha})"
                  style="filter:blur(${this.config.blur}px)"/>
            
            <!-- Effet de bulles liquides -->
            <circle cx="${w * 0.2}" cy="${h * 0.3}" r="${Math.min(w, h) * 0.1}"
                    fill="url(#lg-radial)"/>
            <circle cx="${w * 0.8}" cy="${h * 0.7}" r="${Math.min(w, h) * 0.08}"
                    fill="url(#lg-radial)"/>
            <circle cx="${w * 0.5}" cy="${h * 0.2}" r="${Math.min(w, h) * 0.12}"
                    fill="url(#lg-radial)"/>
        </svg>`;
    }

    /**
     * Encode le SVG en data URI avec optimisation
     */
    svgToDataURI(svgString) {
        // Optimiser le SVG avant d'encoder
        const optimized = svgString
            .replace(/\n/g, '')
            .replace(/>\s+</g, '><')
            .replace(/\s{2,}/g, ' ');
        
        return `data:image/svg+xml,${encodeURIComponent(optimized)}`;
    }

    /**
     * Obtient les dimensions réelles d'une carte
     */
    getCardDimensions(card) {
        const rect = card.getBoundingClientRect();
        const style = getComputedStyle(card);
        const radius = parseFloat(style.borderRadius) || this.config.borderRadius;

        return {
            width: Math.round(Math.max(rect.width, this.config.filterWidth)),
            height: Math.round(Math.max(rect.height, this.config.filterHeight)),
            radius: radius,
        };
    }

    /**
     * Injecte les filtres SVG dans le DOM s'ils n'existent pas
     */
    injectSVGFilters() {
        // Vérifier si les filtres existent déjà
        if (document.getElementById('liquid-glass-method')) {
            return;
        }

        const svgContainer = document.createElement('svg');
        svgContainer.classList.add('svg-filters-liquid-glass');
        svgContainer.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        
        svgContainer.innerHTML = `
            <defs>
                <!-- Filtre principal pour les cartes -->
                <filter id="liquid-glass-method" color-interpolation-filters="sRGB"
                        x="-20%" y="-20%" width="140%" height="140%">
                    
                    <feImage id="method-displacement-image"
                             x="0" y="0" width="100%" height="100%"
                             result="map" preserveAspectRatio="none"/>
                    
                    <!-- CANAL ROUGE (déplacement horizontal) -->
                    <feDisplacementMap in="SourceGraphic" in2="map"
                        id="method-red" xChannelSelector="R" yChannelSelector="G"
                        scale="-120" result="dispRed" />
                    <feColorMatrix in="dispRed" type="matrix"
                        values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0"
                        result="red" />
                    
                    <!-- CANAL VERT (déplacement vertical) -->
                    <feDisplacementMap in="SourceGraphic" in2="map"
                        id="method-green" xChannelSelector="R" yChannelSelector="G"
                        scale="-110" result="dispGreen" />
                    <feColorMatrix in="dispGreen" type="matrix"
                        values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0"
                        result="green" />
                    
                    <!-- CANAL BLEU (distorsion supplémentaire) -->
                    <feDisplacementMap in="SourceGraphic" in2="map"
                        id="method-blue" xChannelSelector="R" yChannelSelector="G"
                        scale="-100" result="dispBlue" />
                    <feColorMatrix in="dispBlue" type="matrix"
                        values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0"
                        result="blue" />
                    
                    <!-- Mélange des canaux (effet d'aberration chromatique) -->
                    <feBlend in="red" in2="green" mode="screen" result="rg" />
                    <feBlend in="rg" in2="blue" mode="screen" result="output" />
                    
                    <!-- Lissage final pour l'effet liquide -->
                    <feGaussianBlur in="output" stdDeviation="0.5" result="blurred" />
                    
                    <!-- Ajuster l'opacité finale -->
                    <feComponentTransfer in="blurred" result="final">
                        <feFuncA type="linear" slope="0.93"/>
                    </feComponentTransfer>
                </filter>
            </defs>
        `;

        document.body.insertBefore(svgContainer, document.body.firstChild);
    }

    /**
     * Applique le displacement à un filtre SVG
     */
    applyDisplacementToFilter(filterSelector, dims) {
        const svg = this.buildDisplacementSVG(dims.width, dims.height, dims.radius);
        const dataUri = this.svgToDataURI(svg);

        // Mettre à jour les feImage
        const feImage = document.querySelector(`${filterSelector} feImage`);
        if (feImage) {
            feImage.setAttribute('href', dataUri);
        }

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
     * Initialise le système
     */
    init() {
        // Injecter les filtres SVG
        this.injectSVGFilters();

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
                    if (entry.target.classList.contains('method-card-liquid')) {
                        this.updateCard(entry.target);
                    }
                }
            });

            this.cards.forEach(card => observer.observe(card));
        }

        // Ajouter les effets d'interaction
        this.addMouseTrackingEffects();

        console.log(`🔮 Liquid Glass Engine: ${this.cards.length} method cards initialized`);
    }

    /**
     * Met à jour une carte spécifique
     */
    updateCard(card) {
        const dims = this.getCardDimensions(card);
        this.applyDisplacementToFilter('#liquid-glass-method', dims);
    }

    /**
     * Met à jour toutes les cartes
     */
    updateAllCards() {
        if (this.cards.length === 0) return;
        
        // Utiliser les dimensions de la première carte pour le filtre
        const firstCard = this.cards[0];
        if (firstCard) {
            this.updateCard(firstCard);
        }
    }

    /**
     * Effet de suivi de souris sur les cartes
     * Crée une rotation 3D subtile qui suit le curseur
     */
    addMouseTrackingEffects() {
        this.cards.forEach(card => {
            card.addEventListener('mousemove', (e) => {
                if (!card.classList.contains('method-card-liquid')) return;

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
                    perspective(1000px)
                    rotateY(${deltaX * 5}deg)
                    rotateX(${-deltaY * 5}deg)
                    translateZ(0)
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
     * Changer la configuration du liquid glass
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        this.updateAllCards();
    }

    /**
     * Obtenir les cartes
     */
    getCards() {
        return this.cards;
    }

    /**
     * Trouver une carte par ID
     */
    getCardById(id) {
        return document.querySelector(`[data-method-id="${id}"]`);
    }
}

// ============================================
// INITIALISATION AUTOMATIQUE
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    // Vérifier si des cartes liquid glass existent
    const liquidCards = document.querySelectorAll('.method-card-liquid');
    
    if (liquidCards.length > 0) {
        // Lancer le moteur liquid glass
        window.liquidGlassEngine = new LiquidGlassCardEngine('.method-card-liquid');
    }
});

// Ré-initialiser si des cartes sont ajoutées dynamiquement
const initLiquidGlass = () => {
    if (window.liquidGlassEngine) {
        window.liquidGlassEngine.cards = document.querySelectorAll('.method-card-liquid');
        window.liquidGlassEngine.updateAllCards();
        window.liquidGlassEngine.addMouseTrackingEffects();
    }
};

// Exporter pour utilisation externe
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { LiquidGlassCardEngine, initLiquidGlass };
}
