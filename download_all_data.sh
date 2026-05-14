#!/bin/bash

echo "🚀 Starting Telecom RAG Dataset Download Pipeline..."

# =========================================================
# CREATE DIRECTORIES
# =========================================================

mkdir -p data/{3gpp_docs,oran_docs,teleqna,oran_datasets,simu5g}
BASE_DIR=$(pwd)

# =========================================================
# HELPER FUNCTION
# =========================================================

download_and_extract () {
    NAME=$1
    URL=$2
    TARGET_DIR=$3

    echo ""
    echo "📥 Downloading: $NAME"

    cd "$TARGET_DIR" || exit

    FILE=$(basename "$URL")

    wget -q --show-progress "$URL" || curl -L -O "$URL"

    if [[ "$FILE" == *.zip ]]; then
        echo "📦 Extracting ZIP: $FILE"
        unzip -o "$FILE" >/dev/null 2>&1
    fi

    if [[ "$FILE" == *.tar.gz ]]; then
        echo "📦 Extracting TAR.GZ: $FILE"
        tar -xzf "$FILE"
    fi

    rm -f "$FILE"

    echo "✅ Finished: $NAME"

    cd "$BASE_DIR" || exit
}

# =========================================================
# 1. DOWNLOAD IMPORTANT 3GPP SPECS
# =========================================================

echo ""
echo "=============================="
echo "📡 DOWNLOADING 3GPP SPECS"
echo "=============================="

declare -A specs=(

    # Core NR Architecture
    ["38.300"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.300/38300-h30.zip"

    # RRC
    ["38.331"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.331/38331-h30.zip"

    # PHY
    ["38.211"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.211/38211-h30.zip"
    ["38.212"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.212/38212-h30.zip"
    ["38.213"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.213/38213-h30.zip"
    ["38.214"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.214/38214-h30.zip"

    # MAC/RLC/PDCP
    ["38.321"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.321/38321-h30.zip"
    ["38.322"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.322/38322-h30.zip"
    ["38.323"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.323/38323-h30.zip"

    # NG-RAN
    ["38.401"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.401/38401-h30.zip"
    ["38.413"]="https://www.3gpp.org/ftp/Specs/archive/38_series/38.413/38413-h30.zip"

    # Dual Connectivity
    ["37.340"]="https://www.3gpp.org/ftp/Specs/archive/37_series/37.340/37340-h30.zip"
)

for spec in "${!specs[@]}"; do
    download_and_extract "$spec" "${specs[$spec]}" "$BASE_DIR/data/3gpp_docs"
done

# =========================================================
# 2. DOWNLOAD TELEQNA
# =========================================================

echo ""
echo "=============================="
echo "📚 DOWNLOADING TELEQNA"
echo "=============================="

cd data/teleqna || exit

git clone https://github.com/netop-team/TeleQnA.git

cd "$BASE_DIR" || exit

echo "✅ TeleQnA downloaded"

# =========================================================
# 3. DOWNLOAD TELE-LLMS
# =========================================================

echo ""
echo "=============================="
echo "🤖 DOWNLOADING TELE-LLMS"
echo "=============================="

cd data || exit

git clone https://github.com/Ali-maatouk/Tele-LLMs.git

cd "$BASE_DIR" || exit

echo "✅ Tele-LLMs downloaded"

# =========================================================
# 4. DOWNLOAD O-RAN DOCS
# =========================================================

echo ""
echo "=============================="
echo "📡 DOWNLOADING O-RAN DOCS"
echo "=============================="

cd data/oran_docs || exit

# Example public O-RAN docs
wget -q --show-progress \
https://public.o-ran.org/display/DOC/O-RAN.WG1.O-RAN-Architecture-Description

wget -q --show-progress \
https://public.o-ran.org/display/DOC/O-RAN.WG3.E2AP

cd "$BASE_DIR" || exit

echo "✅ O-RAN docs downloaded"

# =========================================================
# 5. DOWNLOAD SIMU5G
# =========================================================

echo ""
echo "=============================="
echo "📶 DOWNLOADING SIMU5G"
echo "=============================="

cd data/simu5g || exit

git clone https://github.com/Unipisa/Simu5G.git

cd "$BASE_DIR" || exit

echo "✅ Simu5G downloaded"

# =========================================================
# FINAL CLEANUP
# =========================================================

find data -name "*.zip" -delete
find data -name "__MACOSX" -type d -exec rm -rf {} +

echo ""
echo "================================================="
echo "🎉 ALL TELECOM DATASETS DOWNLOADED SUCCESSFULLY!"
echo "================================================="
echo ""
echo "📂 Directory Structure:"
echo ""
tree data -L 2