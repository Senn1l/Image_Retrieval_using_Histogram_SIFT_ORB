#include "database.h"

// Hàm tạo file CSV với cột name, label, url và histogram
void myDatabase::createCsv_featuresHist(const string& inputCsvFilePath, const string& outputCsvFilePath) {
    ifstream infile(inputCsvFilePath);
    if (!infile.is_open()) {
        cout << "Could not open the file " << inputCsvFilePath << endl;
        return;
    }

    ofstream outfile(outputCsvFilePath);
    if (!outfile.is_open()) {
        cout << "Could not open the file " << outputCsvFilePath << endl;
        infile.close();
        return;
    }

    string line;
    bool isFirstLine = true;
    stringstream buffer;

    // Với mỗi dòng trong infile (CSV)
    while (getline(infile, line)) {
        // Thực hiện phân tách các cột thành 1 vector
        vector<string> fields = split(line, ',');

        if (isFirstLine) {
            // Ghi header
            buffer << "Picture Name,Label,URL,Histogram\n";
            isFirstLine = false;
        }
        else {
            string pictureName = fields[0]; // Tên ảnh
            string buildingName = fields[1]; // Tên tòa nhà (label)
            string imagePath = this->imageFolderPath + "/" + pictureName + ".png";
            Mat histogram = myCalcHistogram(imagePath);
            string histogramStr = histogramToString(histogram);

            buffer << pictureName << "," << buildingName << "," << imagePath << "," << histogramStr << "\n";
        }
    }
    infile.close();

    // Viết buffer vào file
    outfile << buffer.str();

    outfile.close();
    cout << "Updated CSV file saved to " << outputCsvFilePath << endl;
}

// Hàm đọc dữ liệu từ file CSV vào vector myRecord
vector<myRecord> myDatabase::makeRecordsForHist(const string& filePath) {
    vector<myRecord> records;
    ifstream file(filePath);
    if (!file.is_open()) {
        cout << "Error opening file: " << filePath << endl;
        return records;
    }

    string line;
    // Bỏ qua dòng header
    getline(file, line);

    // Thực hiện đọc file csv và lấy ra 4 cột
    while (getline(file, line)) {
        stringstream ss(line);
        string name, label, url, hist_str;
        getline(ss, name, ',');
        getline(ss, label, ',');
        getline(ss, url, ',');
        getline(ss, hist_str, ',');

        // Chuyển đổi chuỗi histogram thành ma trận Mat
        vector<float> histValues;
        istringstream ssHistStr(hist_str);
        string histValue;
        while (getline(ssHistStr, histValue, ';')) {
            histValues.push_back(stof(histValue));
        }
        Mat histogram = Mat(histValues).clone();

        // Tạo một Record mới và thêm vào vector records
        records.push_back({ name, label, url, histogram });
    }
    file.close();
    return records;
}

// Lưu vector descriptors vào file dat (ORB)
void myDatabase::saveDescriptorsToDat(const string& descriptorsFilePath, const vector<Mat>& descriptors) {
    ofstream outFile(descriptorsFilePath, ios::out | ios::binary);
    if (!outFile.is_open()) {
        cerr << "Could not open the file " << descriptorsFilePath << endl;
        return;
    }

    // Ghi số lượng ma trận vào đầu file
    int numDescriptors = int(descriptors.size());
    outFile.write(reinterpret_cast<const char*>(&numDescriptors), sizeof(numDescriptors));

    // Ghi từng ma trận mô tả vào file
    for (const auto& desc : descriptors) {
        // Ghi mã nhận dạng kiểu dữ liệu
        // Nếu là float thì lưu dataType = 1
        int dataType = (desc.type() == CV_32F) ? 1 : 0;
        outFile.write(reinterpret_cast<const char*>(&dataType), sizeof(dataType));

        int rows = desc.rows;
        int cols = desc.cols;
        outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

        // Ghi dữ liệu ma trận
        if (dataType == 1) 
            outFile.write(reinterpret_cast<const char*>(desc.data), rows * cols * sizeof(float));
        else
            outFile.write(reinterpret_cast<const char*>(desc.data), rows * cols * sizeof(uchar));
    }
    outFile.close();
}

// Thực hiện tính sau đó lưu vector descriptors vào file dat, các đường dẫn đến ảnh vào file txt
void myDatabase::create_descriptorsAndURL(const string& descriptorsFilePath, const string& URLfilePath,
    const Size& size, const string& queryType) {
    const Ptr<SIFT> sift = SIFT::create();
    const Ptr<ORB> orb = ORB::create();

    // Xác định sift hay orb
    bool isSift = false;
    if (queryType == "sift") {
        isSift = true;
    }

    vector<string> imagePaths;

    if (!fs::exists(this->imageFolderPath) || !fs::is_directory(this->imageFolderPath)) {
        cout << "Folder does not exist or is not suitable" << endl;
        return;
    }

    ofstream outFile_URL(URLfilePath);
    if (!outFile_URL.is_open()) {
        cout << "Could not open the file " << URLfilePath << endl;
        return;
    }
    stringstream bufferURL;

    vector<Mat> descriptorsList;
    for (const auto& entry : fs::directory_iterator(imageFolderPath)) {
        // Sử dụng buffer lưu image path
        const string image_path = entry.path().string();
        bufferURL << image_path << ",";

        Mat descriptors;
        if (isSift)
            descriptors = computeSIFTDescriptors(sift, image_path, size);
        else
            descriptors = computeORBDescriptors(orb, image_path, size);

        descriptorsList.push_back(descriptors);
        imagePaths.push_back(image_path);
    }
    // Lưu các descriptors vào file .dat
    this->saveDescriptorsToDat(descriptorsFilePath, descriptorsList);

    // Ghi những gì buffer lưu vào file URL
    outFile_URL << bufferURL.str();

    // Đóng file
    outFile_URL.close();
}

// Đọc file dat lấy ra ma trận descriptors và url để lấy đường dẫn
vector<pair<string, Mat>> myDatabase::read_featureAndURL(const string& descriptorsFilePath, const string& URLfilePath) {
    vector<pair<string, Mat>> URL_descriptorsList;

    ifstream inFileURL(URLfilePath);
    if (!inFileURL.is_open()) {
        cout << "Could not open the file " << URLfilePath << endl;
        return {};
    }

    // Lấy các đường dẫn và lưu vào imagePaths
    vector<string> imagePaths;
    string line;
    while (getline(inFileURL, line)) {
        stringstream ss(line);
        string url;
        while (getline(ss, url, ',')) {
            imagePaths.push_back(url);
        }
    }
    inFileURL.close();

    ifstream inFileDescriptors(descriptorsFilePath, ios::in | ios::binary);
    if (!inFileDescriptors.is_open()) {
        cout << "Could not open the file " << descriptorsFilePath << endl;
        return {};
    }

    // Đọc số lượng ma trận mô tả
    int numDescriptors;
    inFileDescriptors.read(reinterpret_cast<char*>(&numDescriptors), sizeof(numDescriptors));
    cout << "Num: " << numDescriptors << endl;

    //cout << dataType;

    // Các imagePaths này được lấy từ tập dữ liệu ảnh nên không có chuyện thiếu
    for (int i = 0; i < numDescriptors; ++i) {
        // Đọc mã nhận dạng kiểu dữ liệu
        int dataType;
        inFileDescriptors.read(reinterpret_cast<char*>(&dataType), sizeof(dataType));

        // Đọc kích thước ma trận
        int rows, cols;
        inFileDescriptors.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        inFileDescriptors.read(reinterpret_cast<char*>(&cols), sizeof(cols));

        Mat descriptors;
        if (dataType == 1) {
            descriptors = Mat(rows, cols, CV_32F); // Kiểu dữ liệu float
            inFileDescriptors.read(reinterpret_cast<char*>(descriptors.data), rows * cols * sizeof(float));
        }
        else {
            descriptors = Mat(rows, cols, CV_8U); // Kiểu dữ liệu uchar
            inFileDescriptors.read(reinterpret_cast<char*>(descriptors.data), rows * cols * sizeof(uchar));
        }

        // Đọc dữ liệu của ma trận
        if (descriptors.empty()) {
            cout << "Failed to read descriptors for image " << i << endl;
            continue;
        }

        URL_descriptorsList.push_back(make_pair(imagePaths[i], descriptors));
    }
    // Đóng file
    inFileDescriptors.close();
    return URL_descriptorsList;
}

// Hàm tạo record từ vector cặp URL và descriptors
vector<myRecord> myDatabase::makeRecords(vector<pair<string, Mat>> URL_descriptorsList, const string& CSVFilePath) {
    // Khởi tạo myRecord
    vector<myRecord> records;

    // Tạo ra một unordered_map lưu trữ key: Picture Name và Value: Label (Building Name)
    unordered_map<string, string> mapNameToLabel = getNameLabelFromCSV(CSVFilePath);

    // Tạo myRecord và thêm vào vector records
    for (const auto& item : URL_descriptorsList) {
        string url = item.first;
        Mat descriptors = item.second;

        // Lấy tên file từ URL
        string fileName = getFileNameFromFilePath(url);
        // Tìm label tương ứng từ unordered_map dựa theo key là tên file
        string label = mapNameToLabel[fileName];

        // Thêm record vào list
        myRecord record;
        record.Name = fileName;
        record.Label = label;
        record.URL = url;
        record.descriptors = descriptors;
        records.push_back(record);
    }
    return records;
}

// Lưu ma trận vào file dat
void myDatabase::saveMatToDat(const string& centersFilePath, const Mat& mat) {
    ofstream outFile(centersFilePath, ios::out | ios::binary);
    if (!outFile.is_open()) {
        cout << "Could not open the file " << centersFilePath << endl;
        return;
    }

    // Ghi từng mô tả ma trận centers vào file
    int rows = mat.rows;
    int cols = mat.cols;
    outFile.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    outFile.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    outFile.write(reinterpret_cast<const char*>(mat.data), mat.rows * mat.cols * sizeof(float));

    // Đóng file
    outFile.close();
}

// Đọc ma trận từ file dat
Mat myDatabase::readMatFromDat(const string& centersFilePath) {
    ifstream inFile(centersFilePath, ios::in | ios::binary);
    if (!inFile.is_open()) {
        cout << "Could not open the file " << centersFilePath << endl;
        return Mat();
    }

    int rows, cols;
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Đọc dữ liệu của ma trận
    Mat centers(rows, cols, CV_32F); // Kiểu dữ liệu unsigned char
    inFile.read(reinterpret_cast<char*>(centers.data), rows * cols * sizeof(float));

    // Đóng file
    inFile.close();
    return centers;
}