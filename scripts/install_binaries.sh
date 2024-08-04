#!/usr/bin/env bash

if [[ -n "$CONDA_PREFIX" ]]; then
	PREFIX=$CONDA_PREFIX
else
	echo "Please activate a conda environment"
	exit 1
fi

if command -v mamba &> /dev/null; then
	CONDA=mamba
else
	CONDA=conda
fi

echo "Installing binaries to $PREFIX"
TEMPDIR=$(mktemp -d)

# Shared objects needed for the cairosvg package
$CONDA install -y -c conda-forge cairo

# Install FiraCode font (for rich export_svg -> cairosvg PDF generation)
if [[ "$OSTYPE" == "darwin"* ]]; then
	brew tap homebrew/cask-fonts
	brew install font-fira-code
else
	FONTDIR="$HOME/.local/share/fonts"
	mkdir -p "$FONTDIR"

	if [ ! command -v fc-list ] &>/dev/null || ! fc-list | grep -q "FiraCode"; then
		echo "FiraCode could not be found. Installing on $FONTDIR"
		# NOTE: we need non-NF FiraCode for python rich export_svg -> cairosvg PDF generation
		mkdir -p "$TEMPDIR/firacode"
		curl -s https://api.github.com/repos/tonsky/FiraCode/releases/latest |
			grep "browser_download_url.*.zip" |
			cut -d : -f 2,3 |
			tr -d \" |
			wget -qi - -O "$TEMPDIR/firacode.zip"
		unzip "$TEMPDIR/firacode.zip" -d "$TEMPDIR"/firacode
		mv "$TEMPDIR"/firacode/ttf/*.ttf "$FONTDIR"

		fc-cache -fv
	else
		echo "FiraCode is already installed"
	fi
fi

rm -rf "$TEMPDIR"
