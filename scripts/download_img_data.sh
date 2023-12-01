
DATASET=$1

echo "Note: available dataset are ffhq, afhqv2 "
echo "Specified [$DATASET]"

if [ "$DATASET" == afhqv2 ]; then
    # afhqv2-64x64.zip
    gdown -O data/ 17mH2b3oQ1CjqNshDBqZWapFoZbl2Rcpy

    # 3x64x64 orthobasis
    gdown -O data/ 1JYm8O46Y2bZOySDghyODp0ud-5lZcKnc

    # network (VP, uncond)
    gdown -O data/ 1A1upwcOmLfcHV0j4n2Fa8rtOnSVcotmh
fi

if [ "$DATASET" == ffhq ]; then
    # ffhq-64x64.zip
    gdown -O data/ 1NQ40CmBxFiAkd2SoWUTbhmm3VAyWVerF

    # 3x64x64 orthobasis
    gdown -O data/ 1JYm8O46Y2bZOySDghyODp0ud-5lZcKnc

    # network (VP, uncond)
    gdown -O data/ 1nz2M3gn-CbQ2z2_oKRSA5M_VwQzrtXdt
fi
