#!/bin/bash


theseFolders="Characters/*";

basePath=$PWD

for thisFolder in $theseFolders
do
    fileName=`basename "$thisFolder"`;
    
    cd $PWD/$thisFolder

    convert -background black -fill white -font $basePath/ZhenHei.ttf -size 28x label:"$fileName" $fileName.png

    cd $basePath

done

