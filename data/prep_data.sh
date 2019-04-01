# assumes single directory 'presidents_all' containing 4 directories (each containing 200 images for named president)

mkdir ./{presidents_150,presidents_100,presidents_50,presidents_25,test}
mkdir ./{presidents_150,presidents_100,presidents_50,presidents_25,test}/{trump,obama,bush,clinton}

for president in trump obama bush clinton
do
ls presidents_all/$president | tail -50 | xargs -I % cp presidents_all/$president/% test/$president/%
ls presidents_all/$president | head -25 | xargs -I % cp presidents_all/$president/% presidents_25/$president/%
ls presidents_all/$president | head -50 | xargs -I % cp presidents_all/$president/% presidents_50/$president/%
ls presidents_all/$president | head -100 | xargs -I % cp presidents_all/$president/% presidents_100/$president/%
ls presidents_all/$president | head -150 | xargs -I % cp presidents_all/$president/% presidents_150/$president/%
done
exit 0
