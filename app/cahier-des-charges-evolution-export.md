# Refonte du système d'exports

## Contexte

Cette analyse concerne l'application de restauration photo située dans `app`.

Je souhaite refondre complètement le système d'exports afin de permettre :

- plusieurs exports par image ;
- la comparaison visuelle des exports ;
- la visualisation et comparaison des paramètres utilisés ;
- la restauration d'anciens paramètres ;
- la suppression d'exports ;
- une gestion entièrement basée sur le répertoire de sortie.

Cette évolution doit fonctionner aussi bien en mode image unique qu'en mode batch.

---

# Mission

Avant toute implémentation :

1. Analyser en détail le fonctionnement actuel.
2. Identifier tous les composants impactés.
3. Identifier les impacts indirects et effets de bord.
4. Poser toutes les questions nécessaires.
5. Proposer une architecture cible propre et maintenable.
6. Fournir un plan d'implémentation détaillé.
7. Définir une stratégie de tests complète.

Ne pas coder immédiatement.

---

# Nouveau modèle d'export

## Comportement général

L'utilisateur continue de choisir un répertoire de sortie.

Lors de chaque export :

- l'image est exportée dans le répertoire de sortie ;
- plusieurs exports peuvent coexister ;
- chaque export possède sa propre recette et ses éventuels masques associés.

Exemple :

```text
photo.export.001.jpg
photo.export.001.recipe.json
photo.export.001.mask.png
photo.export.001.redeye_mask.png

photo.export.002.jpg
photo.export.002.recipe.json
...
```

Le numéro d'export est automatiquement incrémenté.

L'extension du fichier exporté doit toujours rester identique à celle du fichier source.

---

## Recipe

Le fichier recipe :

- est stocké à côté de l'image exportée ;
- utilise le même numéro d'export ;
- contient la configuration complète de l'export ;
- contient la date/heure d'export.

---

## Suppression de result.json

Supprimer complètement `result.json`.

Ne plus conserver :

- timing ;
- step_log ;
- autres données inutiles.

La date/heure d'export doit être intégrée au recipe.

---

# Refonte de l'onglet Résultat

L'onglet Résultat devient le centre de gestion des exports.

## Vue mosaïque

Afficher tous les exports disponibles pour l'image courante.

Objectifs :

- présentation propre et moderne ;
- adaptation automatique au nombre d'exports ;
- chargement efficace ;
- navigation fluide.

---

## Synchronisation de la vue

Toutes les vues doivent partager :

- zoom ;
- position ;
- déplacement.

Cela inclut :

- image source ;
- image restaurée ;
- mosaïque d'exports ;
- export agrandi.

Le changement d'onglet ne doit jamais réinitialiser la vue.

La seule exception est le changement d'image dans le batch.

---

## Agrandissement

Double-clic sur un export :

- affichage plein format ;
- conservation exacte du zoom et de la position ;
- même zone d'affichage que les autres onglets pour faciliter les comparaisons (pas de texte en dessous ou au dessus qui décalerait l'image)

Le passage mosaïque ↔ plein format doit être transparent.

---

# Panneau latéral Export

La sélection d'un export affiche un panneau latéral contenant :

- informations de l'export ;
- date ;
- paramètres utilisés ;
- comparaison visuelle avec les autres exports ;
- actions.

Actions :

- Restaurer les paramètres ;
- Supprimer l'export.

---

# Comparaison des paramètres

Supprimer :

- l'ancien affichage des paramètres en lecture seule;
- l'onglet Diff ;
- le diff textuel.

Créer une nouvelle visualisation compacte adaptée à un panneau latéral.

Objectifs :

- voir immédiatement quelles étapes sont activées ;
- voir quelles étapes diffèrent d'au moins un autre export ;
- mettre clairement en évidence les différences.

Par défaut :

- afficher uniquement les paramètres différents.

Ajouter un bouton :

```text
Afficher tous les paramètres
```

qui affiche également les paramètres communs.

Les différences doivent toujours être mises en évidence visuellement.

---

# Suppression d'un export

Depuis le panneau latéral :

- demander confirmation ;
- supprimer :
  - image exportée ;
  - recipe ;
  - mask ;
  - redeye_mask ;
  - tous les fichiers associés.

---

# Mode image unique

Même comportement que le batch.

Ctrl+S :

- première utilisation : demande le répertoire de sortie ;
- ensuite : export automatique avec incrément du numéro d'export.

---

# Contraintes d'architecture

La solution doit :

- rester simple ;
- être modulaire ;
- être facilement testable ;
- éviter les duplications entre mode single et batch ;
- conserver un code propre et maintenable.

Identifier explicitement :

- les nouveaux modèles ;
- les services à créer ;
- les composants UI à créer ;
- les mécanismes de synchronisation de vue ;
- les points de mutualisation entre batch et single.

---

# Livrables attendus

Avant toute implémentation :

## Analyse de l'existant

- flux actuels ;
- composants concernés ;
- dépendances.

## Questions ouvertes

Lister tous les points nécessitant une décision.

## Architecture cible

- modèles ;
- services ;
- persistance ;
- UI ;
- synchronisation des vues.

## Plan d'implémentation

Découpage détaillé en étapes indépendantes.

## Plan de tests

Inclure :

- tests unitaires ;
- tests d'intégration ;
- tests UI ;
- scénarios utilisateur complets.

Enfin, identifier tout impact ou besoin auquel je n'aurais pas pensé concernant :
- la gestion multi-exports ;
- les performances ;
- l'ergonomie ;
- la suppression ;
- la restauration des paramètres ;
- la synchronisation des vues ;
- le mode batch ;
- le mode image unique.