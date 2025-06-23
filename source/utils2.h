#pragma once
#include "libs.h"

#include "localFeature.h"
#include "globalFeature.h"
#include "database.h"

void processGlobalFeature(const string& queryType, const string& imageFolderPath,
    const string& descriptorsFilePath, const string& csvFilePath);

void processLocalFeature(const string& queryType, const string& imageFolderPath, const string& descriptorsFilePath,
    const string& URLFilePath, const string& csvFilePath, const Size& sizeUsing);

void processLocalFeatureKmeans(const string& queryType, const string& imageFolderPath, const string& descriptorsFilePath,
    const string& URLFilePath, const string& csvFilePath, const string& centersFilePath, const Size& sizeUsing, int clusterCount);