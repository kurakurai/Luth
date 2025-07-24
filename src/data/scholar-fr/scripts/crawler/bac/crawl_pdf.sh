#!/bin/bash

BASE_URL="https://www.sujetdebac.fr/"
PDF_FILE="pdf_urls.txt"

declare -A visited_depths
pdf_urls=()

touch "$PDF_FILE"

function normalize_url() {
  local url="$1"
  # Supprime la partie après #
  echo "${url%%#*}"
}

function add_pdf_url() {
  local url
  url=$(normalize_url "$1")
  if ! grep -Fxq "$url" "$PDF_FILE"; then
    echo "$url" >> "$PDF_FILE"
    echo "Added PDF URL: $url"
  fi
}

function crawl() {
  local url
  url=$(normalize_url "$1")
  local depth=$2

  local prev_depth=${visited_depths[$url]:-9999}

  # Si on a déjà visité à une profondeur plus faible ou égale, on ne recrawl pas
  if (( depth >= prev_depth )); then
    return
  fi

  # Vérifie que l'URL commence bien par BASE_URL
  if [[ "$url" != "$BASE_URL"* ]]; then
    return
  fi

  # Ignorer fichiers mp3, mp4, avi
  if [[ "$url" =~ \.(mp3|mp4|avi)$ ]]; then
    echo "Ignoring media file: $url"
    return
  fi

  visited_depths[$url]=$depth

  echo "Crawling $url (depth $depth)..."

  # Récupérer les liens
  mapfile -t links < <(lynx -listonly -dump "$url" | tail -n +3 | awk '{$1=""; print $0}' | sed 's/^ *//')

  for link in "${links[@]}"; do
    link=$(normalize_url "$link")

    # Filtrer seulement les liens du domaine cible
    if [[ "$link" != "$BASE_URL"* ]]; then
      continue
    fi

    # Ignorer fichiers mp3, mp4, avi
    if [[ "$link" =~ \.(mp3|mp4|avi)$ ]]; then
      echo "Ignoring media file link: $link"
      continue
    fi

    if [[ "$link" =~ \.pdf$ ]]; then
      if [[ ! " ${pdf_urls[*]} " =~ " $link " ]]; then
        pdf_urls+=("$link")
        add_pdf_url "$link"
      fi
    else
      crawl "$link" $((depth + 1))
    fi
  done
}

crawl "$BASE_URL" 0

echo "Crawl terminé."
echo "Liste des PDF sauvegardée dans $PDF_FILE"

