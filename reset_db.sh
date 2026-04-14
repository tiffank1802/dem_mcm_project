#!/bin/bash
# Script pour vider et recharger la base de données de DEM_MCM

cd /teamspace/studios/this_studio/dem_mcm_project

echo "════════════════════════════════════════════════════════════"
echo "🔄 RÉINITIALISATION DE LA BASE DE DONNÉES"
echo "════════════════════════════════════════════════════════════"
echo ""

# Vérifier que reset_db.py existe
if [ ! -f "reset_db.py" ]; then
    echo "❌ Erreur: reset_db.py non trouvé dans $(pwd)"
    exit 1
fi

# Mode d'exécution
MODE=${1:-full}

echo "📋 Mode: $MODE"
echo ""

if [ "$MODE" = "dry-run" ]; then
    echo "🔍 Mode dry-run - affichage des dossiers sans importer..."
    python reset_db.py --dry-run
elif [ "$MODE" = "empty" ]; then
    echo "🗑️  Vidage de la DB (sans recharger)..."
    python reset_db.py --empty
else
    echo "⚠️  ATTENTION: Cela va VIDER et RECHARGER la base de données!"
    echo "   Les anciennes données seront perdues."
    echo ""
    read -p "Êtes-vous sûr? (taper 'oui' pour confirmer): " confirm
    
    if [ "$confirm" = "oui" ]; then
        echo ""
        python reset_db.py --full
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "✅ RÉINITIALISATION TERMINÉE"
        echo "════════════════════════════════════════════════════════════"
    else
        echo "❌ Opération annulée"
        exit 1
    fi
fi
