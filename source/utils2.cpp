#include "utils2.h"

void processGlobalFeature(const string& queryType, const string& imageFolderPath, 
    const string& descriptorsFilePath, const string& csvFilePath) {
    // 1. Kiểm tra đường dẫn folder ảnh hợp lệ
    if (!fs::exists(imageFolderPath) || !fs::is_directory(imageFolderPath)) {
        cout << "Image folder does not exist or is not suitable" << endl;
        return;
    }
    // 1.1 Khởi tạo database
    myDatabase db(imageFolderPath);

    // 2. Mở kiểm tra đã có file đặc trưng chưa
    ifstream file1(descriptorsFilePath);
    // Nếu có prompt người dùng có xử lý lại file không
    cout << "=== Creating feature file ===" << endl;
    if (file1.good()) {
        char input;
        cout << "File " << descriptorsFilePath << " exists" << endl;
        cout << "Continue to create file? - (Y/N): ";
        cin >> input;
        char choice = tolower(input);
        if (choice == 'y') {
            auto begin_mkFile = high_resolution_clock::now();
            db.createCsv_featuresHist(csvFilePath, descriptorsFilePath);
            auto end_mkFile = high_resolution_clock::now();
            auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
            cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
        }
    }
    else {
        auto begin_mkFile = high_resolution_clock::now();
        db.createCsv_featuresHist(csvFilePath, descriptorsFilePath);
        auto end_mkFile = high_resolution_clock::now();
        auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
        cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
    }
    cin.ignore(); // Bỏ ký tự '\n'
    file1.close();

    // 3. Sau khi hoàn thành tạo file, tiếp theo là đọc file tạo thành myRecord
    cout << "=== Reading feature file ===" << endl;
    vector<myRecord> records;
    auto begin_read = high_resolution_clock::now();
    records = db.makeRecordsForHist(descriptorsFilePath);
    auto end_read = high_resolution_clock::now();
    auto duration_read = duration_cast<milliseconds>(end_read - begin_read);
    cout << "Read descriptor file completed, duration: " << duration_read.count() << "ms" << endl;

    // 4. Truy vấn ảnh
    cout << "=== Retrieval phase ===" << endl;
    // 4.1 Khởi tạo Factory
    GlobalFeatureFactory GFF;
    // 4.2 Tạo đối tượng với records sử dụng GlobalFeatureFactory
    GlobalFeature* globalFeature = GFF.getFeature(queryType, records);
    // 4.3 Cho người dùng truy vấn qua vòng lặp
    vector<pair<string, double>> topNImagesPath; // lưu kết quả truy vấn
    while (true) {
        // 4.3.1 Nhập Query Path
        cout << "-- Input query image path: ";
        string queryImagePath; //Đường dẫn đến ảnh truy vấn
        getline(cin, queryImagePath);
        // 4.3.1.1 Thực hiện kiểm tra mở được ảnh không?
        Mat image = imread(queryImagePath, IMREAD_COLOR);
        if (image.empty()) {
            cout << "Could not open image! Path: " << queryImagePath << endl;
            continue;
        }
        // 4.3.2 Nhập số lượng ảnh trả về
        cout << "-- Input numbers of image to return (default = 3, min = 3, max = 21): ";
        int n = 3;
        cin >> n;
        cin.ignore();
        if (n < 3) n = 3;
        if (n > 21) n = 21;
        // 4.3.3 Thực hiện truy vấn
        if (globalFeature) {
            cout << "Searching, please wait..." << endl;
            auto begin_search = high_resolution_clock::now();
            topNImagesPath = globalFeature->searchGlobal(queryImagePath, n);
            auto end_search = high_resolution_clock::now();
            auto duration_search = duration_cast<milliseconds>(end_search - begin_search);
            cout << "Search Completed, duration: " << duration_search.count() << "ms\n";
        }
        else
            cout << "[Debugging] Failed to create feature" << endl;
        // 4.3.4 Hiển thị độ tương đồng
        for (const auto& pair : topNImagesPath) {
            cout << "Image Path: " << pair.first << ", Similarity: " << pair.second << endl;
        }
        // 4.3.5 Chuyển đổi vector cặp về vector ma trận sau khi truy vấn
        vector<Mat> topNImages = myCvt_pair_to_Mat(topNImagesPath);
        // 4.3.6 Hiển thị ảnh
        string namedWindow = queryType + "_top " + to_string(n) + " results";
        showImages(topNImages, namedWindow);

        // 4.4 Điều kiện dừng
        cout << "-- Do you want to continue to query? (Y/N) - ";
        char input;
        cin >> input;
        cin.ignore();
        char choice = tolower(input);
        if (choice != 'y') {
            break;
        }
    }
    
    // 5. Ghi MAP
    // 5.1 Cho người dùng nhập có muốn ghi MAP không
    cout << "-- Do you want to write MAP? (Y/N) - ";
    char input;
    cin >> input;
    cin.ignore();
    char choice = tolower(input);
    if (choice != 'y') {
        // Kết thúc chương trình, xóa con trỏ
        if (globalFeature) delete globalFeature;
        cout << "Exitting..." << endl;
        return;
    }
    // 5.2 Lấy đường dẫn file MAP
    string MAPcsvPath;
    cout << "-- Input MAP path (must be a .csv file): ";
    getline(cin, MAPcsvPath);
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        MAPcsvPath = queryType + "_MAP.csv";
        cout << "Assigning MAP path to " << MAPcsvPath << endl;
    }
    // 5.3 Lấy số dòng tối đa ghi vào file (sửa ở đây để ghi map nhiều)
    int maxRows = 5;
    cout << "-- Input maxRows (default = 5, min = 1, max = 1118): ";
    cin >> maxRows;
    cin.ignore();
    if (maxRows < 1) maxRows = 1;
    if (maxRows > 1118) maxRows = 1118;
    // 5.4 Thực hiện ghi MAP
    cout << "Starting Writing MAPs into " << MAPcsvPath << endl;
    globalFeature->processWriteMap(MAPcsvPath, maxRows);

    // 6. Khi kết thúc chương trình, xóa con trỏ
    if (globalFeature) delete globalFeature;
    return;
}

void processLocalFeature(const string& queryType, const string& imageFolderPath, const string& descriptorsFilePath, 
    const string& URLFilePath, const string& csvFilePath, const Size& sizeUsing) {
    // 1. Kiểm tra đường dẫn folder ảnh hợp lệ
    if (!fs::exists(imageFolderPath) || !fs::is_directory(imageFolderPath)) {
        cout << "Image folder does not exist or is not suitable" << endl;
        return;
    }
    // 1.1 Khởi tạo database
    myDatabase db(imageFolderPath);

    // 2. Mở kiểm tra đã có file đặc trưng và URL chưa
    ifstream file1(descriptorsFilePath);
    ifstream file2(URLFilePath);
    // Nếu có prompt người dùng có xử lý file không
    cout << "=== Creating feature file ===" << endl;
    if (file1.good() && file2.good()) {
        char input;
        cout << "File " << descriptorsFilePath << " exists" << endl;
        cout << "Continue to create file? - (Y/N): ";
        cin >> input;
        char choice = tolower(input);
        if (choice == 'y') {
            auto begin_mkFile = high_resolution_clock::now();
            db.create_descriptorsAndURL(descriptorsFilePath, URLFilePath, sizeUsing, queryType);
            auto end_mkFile = high_resolution_clock::now();
            auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
            cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
        }
    }
    else {
        auto begin_mkFile = high_resolution_clock::now();
        db.create_descriptorsAndURL(descriptorsFilePath, URLFilePath, sizeUsing, queryType);
        auto end_mkFile = high_resolution_clock::now();
        auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
        cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
    }
    cin.ignore(); // bỏ ký tự '\n'
    file1.close();
    file2.close();

    // 3. Sau khi hoàn thành tạo file, tiếp theo là đọc file tạo thành myRecord
    cout << "=== Reading feature file ===" << endl;
    vector<myRecord> records;
    // 3.1 SIFT và ORB cần đi qua bước trung gian là tạo ra vector cặp (string, Mat): tương ứng URL và Descriptor
    auto begin_read = high_resolution_clock::now();
    vector<pair<string, Mat>> URL_descriptorsList = db.read_featureAndURL(descriptorsFilePath, URLFilePath);
    // 3.2 Xử lý vector cặp, đồng thời đọc file csvFilePath lấy ra cả label chuyển thành myRecord
    records = db.makeRecords(URL_descriptorsList, csvFilePath);
    auto end_read = high_resolution_clock::now();
    auto duration_read = duration_cast<milliseconds>(end_read - begin_read);
    cout << "Read descriptors and URL files completed, duration: " << duration_read.count() << "ms" << endl;

    // 4. Truy vấn ảnh
    cout << "=== Retrieval phase ===" << endl;
    // 4.1 Khởi tạo Factory
    LocalFeatureFactory LFF;
    // 4.2 Tạo đối tượng với records sử dụng GlobalFeatureFactory
    LocalFeature* localFeature = LFF.getFeature(queryType, records);
    // 4.3 Cho người dùng truy vấn qua vòng lặp
    vector<pair<string, double>> topNImagesPath; // lưu kết quả truy vấn
    while (true) {
        // 4.3.1 Nhập Query Path
        cout << "-- Input query image path: ";
        string queryImagePath; //Đường dẫn đến ảnh truy vấn
        getline(cin, queryImagePath);
        // Thực hiện kiểm tra mở được ảnh không?
        Mat image = imread(queryImagePath, IMREAD_COLOR);
        if (image.empty()) {
            cout << "Could not open image! Path: " << queryImagePath << endl;
            continue;
        }
        // 4.3.2 Nhập số lượng ảnh trả về
        cout << "-- Input numbers of image to return (default = 3, min = 3, max = 21): ";
        int n = 3;
        cin >> n;
        cin.ignore();
        if (n < 3) n = 3;
        if (n > 21) n = 21;
        // 4.3.3 Thực hiện truy vấn
        if (localFeature) {
            cout << "Searching, please wait..." << endl;
            auto begin_search = high_resolution_clock::now();
            topNImagesPath = localFeature->searchLocal(queryImagePath, n, sizeUsing);
            auto end_search = high_resolution_clock::now();
            auto duration_search = duration_cast<milliseconds>(end_search - begin_search);
            cout << "Search Completed, duration: " << duration_search.count() << "ms\n";
        }
        else
            cout << "[Debugging] Failed to create feature" << endl;
        // 4.3.4 Hiển thị độ tương đồng
        for (const auto& pair : topNImagesPath) {
            cout << "Image Path: " << pair.first << ", Similarity: " << pair.second << endl;
        }
        // 4.3.5 Chuyển đổi vector cặp về vector ma trận sau khi truy vấn
        vector<Mat> topNImages = myCvt_pair_to_Mat(topNImagesPath);
        // 4.3.6 Hiển thị ảnh
        string namedWindow = queryType + "_top " + to_string(n) + " results";
        showImages(topNImages, namedWindow);

        // 4.4 Điều kiện dừng
        cout << "-- Do you want to continue to query? (Y/N) - ";
        char input;
        cin >> input;
        cin.ignore();
        char choice = tolower(input);
        if (choice != 'y') {
            break;
        }
    }

    // 5. Ghi MAP
    // 5.1 Cho người dùng nhập có muốn ghi MAP không
    cout << "-- Do you want to write MAP? (Y/N) - ";
    char input;
    cin >> input;
    cin.ignore();
    char choice = tolower(input);
    if (choice != 'y') {
        // Kết thúc chương trình, xóa con trỏ
        if (localFeature) delete localFeature;
        cout << "Exitting..." << endl;
        return;
    }
    // 5.2 Lấy đường dẫn file MAP
    string MAPcsvPath;
    cout << "-- Input MAP path (must be a .csv file): ";
    getline(cin, MAPcsvPath);
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        MAPcsvPath = queryType + "_MAP.csv";
        cout << "Assigning MAP path to " << MAPcsvPath << endl;
    }
    // 5.3 Lấy số dòng tối đa ghi vào file (sửa ở đây để ghi map nhiều)
    int maxRows = 5;
    cout << "-- Input maxRows (default = 5, min = 1, max = 1118): ";
    cin >> maxRows;
    cin.ignore();
    if (maxRows < 1) maxRows = 1;
    if (maxRows > 1118) maxRows = 1118;
    // 5.4 Thực hiện ghi MAP
    cout << "Starting Writing MAPs into " << MAPcsvPath << endl;
    localFeature->processWriteMap(MAPcsvPath, sizeUsing, maxRows);

    // 6. Khi kết thúc chương trình, xóa con trỏ
    if (localFeature) delete localFeature;
    return;
}

void processLocalFeatureKmeans(const string& queryType, const string& imageFolderPath, const string& descriptorsFilePath,
    const string& URLFilePath, const string& csvFilePath, const string& centersFilePath, const Size& sizeUsing, int clusterCount) {

    // 1. Kiểm tra đường dẫn folder ảnh hợp lệ
    if (!fs::exists(imageFolderPath) || !fs::is_directory(imageFolderPath)) {
        cout << "Image folder does not exist or is not suitable" << endl;
        return;
    }
    // 1.1 Khởi tạo database
    myDatabase db(imageFolderPath);

    // 2. Mở kiểm tra đã có file đặc trưng chưa
    ifstream file1(descriptorsFilePath);
    ifstream file2(URLFilePath);
    // Nếu có prompt người dùng có xử lý file không
    cout << "=== Creating feature file ===" << endl;
    cout << "Please wait..." << endl;
    if (file1.good() && file2.good()) {
        char input;
        cout << "File " << descriptorsFilePath << " exists" << endl;
        cout << "Continue to create file? - (Y/N): ";
        cin >> input;
        char choice = tolower(input);
        if (choice == 'y') {
            auto begin_mkFile = high_resolution_clock::now();
            db.create_descriptorsAndURL(descriptorsFilePath, URLFilePath, sizeUsing, queryType);
            auto end_mkFile = high_resolution_clock::now();
            auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
            cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
        }
    }
    else {
        auto begin_mkFile = high_resolution_clock::now();
        db.create_descriptorsAndURL(descriptorsFilePath, URLFilePath, sizeUsing, queryType);
        auto end_mkFile = high_resolution_clock::now();
        auto duration_mkFile = duration_cast<milliseconds>(end_mkFile - begin_mkFile);
        cout << "Create file(s) completed, duration: " << duration_mkFile.count() << "ms" << endl;
    }
    cin.ignore(); //bỏ ký tự '\n'
    file1.close();
    file2.close();

    // 3. Sau khi hoàn thành tạo file, tiếp theo là đọc file tạo thành myRecord
    cout << "=== Reading feature file ===" << endl;
    vector<myRecord> records;
    // 3.1 SIFT và ORB cần đi qua bước trung gian là tạo ra vector cặp (string, Mat): tương ứng URL và Descriptor
    auto begin_read = high_resolution_clock::now();
    vector<pair<string, Mat>> URL_descriptorsList = db.read_featureAndURL(descriptorsFilePath, URLFilePath);
    // 3.2 Xử lý vector cặp, đồng thời đọc file csvFilePath lấy ra cả label chuyển thành myRecord
    records = db.makeRecords(URL_descriptorsList, csvFilePath);
    auto end_read = high_resolution_clock::now();
    auto duration_read = duration_cast<milliseconds>(end_read - begin_read);
    cout << "Read descriptors and URL files completed, duration: " << duration_read.count() << "ms" << endl;

    // 4. Áp dụng kmeans
    // Khởi tạo đối tượng sift với records
    cout << "=== Kmeans phase ===" << endl;
    // 4.1 Khởi tạo Factory
    LocalFeatureFactory LFF;
    // 4.2 Tạo đối tượng với records
    LocalFeature* localFeature = LFF.getFeature(queryType, records);
    // 4.3 Làm codebook và lưu
    Mat codebook;
    ifstream file3(centersFilePath);
    // Nếu có prompt người dùng có xử lý file không
    cout << "=== Creating codebook file ===" << endl;
    cout << "Making code book, please wait..." << endl;
    if (file3.good()) {
        char input;
        cout << "File " << centersFilePath << " exists" << endl;
        cout << "Continue to create file? - (Y/N): ";
        cin >> input;
        char choice = tolower(input);
        if (choice == 'y') {
            auto begin_mkCb = high_resolution_clock::now();
            // Làm codebook
            codebook = localFeature->createCodebook(clusterCount);
            // Lưu codebook
            db.saveMatToDat(centersFilePath, codebook);
            auto end_mkCb = high_resolution_clock::now();
            auto duration_mkCb = duration_cast<milliseconds>(end_mkCb - begin_mkCb);
            cout << "Create and save codebook completed, duration: " << duration_mkCb.count() << "ms\n";
        }
    }
    else {
        auto begin_mkCb = high_resolution_clock::now();
        // Làm codebook
        codebook = localFeature->createCodebook(clusterCount);
        // Lưu codebook
        db.saveMatToDat(centersFilePath, codebook);
        auto end_mkCb = high_resolution_clock::now();
        auto duration_mkCb = duration_cast<milliseconds>(end_mkCb - begin_mkCb);
        cout << "Create and save codebook completed, duration: " << duration_mkCb.count() << "ms\n";
    }
    cin.ignore(); //Bỏ ký tự '\n'
    file3.close();
    // 4.4 Đọc codebook
    codebook = db.readMatFromDat(centersFilePath);
    // 4.5 Kiểm tra theo query type để thực hiện chuyển đổi codebook sang dạng CV_8U (binary) nếu là orb
    if (queryType == "orb") {
        codebook.convertTo(codebook, CV_8U);
    }
    // 4.6 Lấy ra histogram (tần số codeword) từ codebook
    auto begin_mkHistCb = high_resolution_clock::now();
    vector<Mat> histograms = localFeature->createHistograms(codebook);
    auto end_mkHistCb = high_resolution_clock::now();
    auto duration_mkHistCb = duration_cast<milliseconds>(end_mkHistCb - begin_mkHistCb);
    cout << "Make Bag of Words completed, duration: " << duration_mkHistCb.count() << "ms\n";

    // 5. Truy vấn ảnh
    cout << "=== Retrieval phase ===" << endl;
    vector<pair<string, double>> topNImagesPath; // lưu kết quả truy vấn
    while (true) {
        // 5.1 Nhập Query Path
        cout << "-- Input query image path: ";
        string queryImagePath; //Đường dẫn đến ảnh truy vấn
        getline(cin, queryImagePath);
        // Thực hiện kiểm tra mở được ảnh không?
        Mat image = imread(queryImagePath, IMREAD_COLOR);
        if (image.empty()) {
            cout << "Could not open image! Path: " << queryImagePath << endl;
            continue;
        }
        // 5.2 Nhập số lượng ảnh trả về
        cout << "-- Input numbers of image to return (default = 3, min = 3, max = 21): ";
        int n = 3;
        cin >> n;
        cin.ignore();
        if (n < 3) n = 3;
        if (n > 21) n = 21;
        // 5.3 Thực hiện truy vấn
        if (localFeature) {
            cout << "Searching, please wait..." << endl;
            auto begin_search = high_resolution_clock::now();
            topNImagesPath = localFeature->searchLocalKmeans(queryImagePath, n, sizeUsing, codebook, histograms);
            auto end_search = high_resolution_clock::now();
            auto duration_search = duration_cast<milliseconds>(end_search - begin_search);
            cout << "Search Completed, duration: " << duration_search.count() << "ms\n";
        }
        else
            cout << "[Debugging] Failed to create feature" << endl;
        // 5.4 Hiển thị độ tương đồng
        for (const auto& pair : topNImagesPath) {
            cout << "Image Path: " << pair.first << ", Similarity: " << pair.second << endl;
        }
        // 5.5 Chuyển đổi vector cặp về vector ma trận sau khi truy vấn
        vector<Mat> topNImages = myCvt_pair_to_Mat(topNImagesPath);
        // 5.6 Hiển thị ảnh
        string namedWindow = queryType + "_top " + to_string(n) + " results";
        showImages(topNImages, namedWindow);
        // 5.7 Điều kiện dừng
        cout << "-- Do you want to continue to query? (Y/N) - ";
        char input;
        cin >> input;
        cin.ignore();
        char choice = tolower(input);
        if (choice != 'y') {
            break;
        }
    }

    // 6. Ghi MAP
    // 6.1 Cho người dùng nhập có muốn ghi MAP không
    cout << "-- Do you want to write MAP? (Y/N) - ";
    char input;
    cin >> input;
    cin.ignore();
    char choice = tolower(input);
    if (choice != 'y') {
        // Kết thúc chương trình
        if (localFeature) delete localFeature;
        cout << "Exitting..." << endl;
        return;
    }
    // 6.2 Lấy đường dẫn file MAP
    string MAPcsvPath;
    cout << "-- Input MAP path (must be a .csv file): ";
    getline(cin, MAPcsvPath);
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        MAPcsvPath = queryType + "_MAP.csv";
        cout << "Assigning MAP path to " << MAPcsvPath << endl;
    }
    // 6.3 Lấy số dòng tối đa ghi vào file
    int maxRows = 5;
    cout << "-- Input maxRows (default = 5, min = 1, max = 1118): ";
    cin >> maxRows;
    cin.ignore();
    if (maxRows < 1) maxRows = 1;
    if (maxRows > 1118) maxRows = 1118;
    // 6.4 Thực hiện ghi MAP
    cout << "Starting Writing MAPs into " << MAPcsvPath << endl;
    localFeature->processWriteMap_Kmeans(MAPcsvPath, sizeUsing, maxRows, codebook, histograms);

    if (localFeature) delete localFeature;
    return;
}