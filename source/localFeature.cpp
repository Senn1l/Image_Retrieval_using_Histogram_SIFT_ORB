#include "localFeature.h"

// Truy vấn Sift với record đã tạo
vector<pair<string, double>> SIFTFeature::searchLocal(const string& queryImagePath, int n, const Size& size) {
    const Ptr<SIFT> sift = SIFT::create();
    Mat queryDescriptors = computeSIFTDescriptors(sift, queryImagePath, size);

    // Khởi tạo matcher
    //FlannBasedMatcher matcher(makePtr<flann::KDTreeIndexParams>(3));
    BFMatcher matcher(NORM_L2, false);
    vector<DMatch> matches_query_current;

    vector<pair<string, double>> imageSimilarities; // Lưu trữ đường dẫn ảnh và độ tương đồng là khoảng cách Euclide

    // Duyệt qua mỗi record lấy descriptor và thực hiện truy vấn
    for (const auto& r : this->records) {
        // Bắt đầu matching
        matcher.match(queryDescriptors, r.descriptors, matches_query_current);

        // Kiểm tra xem có khớp nào được tìm thấy không
        if (matches_query_current.empty()) {
            cout << "No matches found between query and image " << r.Name << endl;
            // Pushback một distance lớn vào
            imageSimilarities.push_back(make_pair(r.URL, 1e6));
            continue;
        }

        // Tính tổng khoảng cách của các matches
        double distance = 0.0;
        for (const auto& item : matches_query_current) {
            distance += item.distance;
        }
        imageSimilarities.push_back(make_pair(r.URL, distance));
    }

    // Sắp xếp vector theo độ tương đồng tăng dần
    // Càng nhỏ thì ảnh càng giống (Khoảng cách Euclide)
    sort(imageSimilarities.begin(), imageSimilarities.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second < b.second;
        });

    // Chọn ra n ảnh
    if (imageSimilarities.size() > n) {
        imageSimilarities.resize(n);
    }

    //for (const auto& pair : imageSimilarities) {
    //    cout << "Image Path: " << pair.first << ", Euclide Distance: " << pair.second << endl;
    //}

    return imageSimilarities;
}

vector<pair<string, double>> ORBFeature::searchLocal(const string& queryImagePath, int n, const Size& size) {
    const Ptr<ORB> orb = ORB::create();
    Mat queryDescriptors = computeORBDescriptors(orb, queryImagePath, size);

    // Khởi tạo matcher
    BFMatcher matcher(NORM_HAMMING, false);
    vector<DMatch> matches_query_current;

    vector<pair<string, double>> imageSimilarities; // Lưu trữ đường dẫn ảnh và độ tương đồng là khoảng cách Euclide

    // Duyệt qua mỗi record lấy descriptor và thực hiện truy vấn
    for (const auto& r : records) {
        // Bắt đầu matching
        matcher.match(queryDescriptors, r.descriptors, matches_query_current);

        // Kiểm tra xem có khớp nào được tìm thấy không
        if (matches_query_current.empty()) {
            cout << "No matches found between query and image " << r.Name << endl;
            // Pushback một distance lớn vào
            imageSimilarities.push_back(make_pair(r.URL, 1e6));
            continue;
        }

        // Tính tổng khoảng cách của các matches
        double distance = 0.0;
        for (const auto& item : matches_query_current) {
            distance += item.distance;
        }
        imageSimilarities.push_back(make_pair(r.URL, distance));
    }

    // Sắp xếp vector theo độ tương đồng tăng dần
    // Càng nhỏ thì ảnh càng giống (Khoảng cách Euclide)
    sort(imageSimilarities.begin(), imageSimilarities.end(), [](const pair<string, double>& a, const pair<string, double>& b) {
        return a.second < b.second;
        });

    // Chọn ra n ảnh
    if (imageSimilarities.size() > n) {
        imageSimilarities.resize(n);
    }

    //for (const auto& pair : imageSimilarities) {
    //    cout << "Image Path: " << pair.first << ", Euclide Distance: " << pair.second << endl;
    //}

    return imageSimilarities;
}

// Tổng hợp n = 3, 5, 11, 21 ghi lại MAP vào một file CSV
// Có các cột là: URL,N=3,N=5,N=11,N=21
void SIFTFeature::computeAndWriteMAP(const string& queryImagePath, const Size& size, ofstream& outputFile) {
    // 1. Khởi tạo APList để lưu vào và list returnImages
    vector<int> numberOfImagesList = { 3, 5, 11, 21 };
    vector<float> APs;

    // 2. Lấy label theo queryImage trong csvFile
    string queryLabel;
    bool foundQueryLabel = false;

    // 2.1 Tìm kiếm queryLabel trong records
    for (const auto& record : this->records) {
        if (record.URL == queryImagePath) {
            queryLabel = record.Label;
            foundQueryLabel = true;
            break;
        }
    }

    // 2.2 Không tìm thấy label trong file CSV
    if (!foundQueryLabel) {
        cout << "Query label not found in CSV file." << endl;
        return;
    }

    // 3. Thực hiện truy vấn 1 lần duy nhất với 21 tấm ảnh vì lúc nào cũng trả ra top 21 tấm
    vector<pair<string, double>> top21imagePaths = this->searchLocal(queryImagePath, 22, size);
    // Xóa ảnh đầu tiên (vì ảnh đầu tiên luôn là ảnh query)
    if (!top21imagePaths.empty() && top21imagePaths.front().first == queryImagePath) {
        top21imagePaths.erase(top21imagePaths.begin());
    }
    // Lấy top 3 ảnh
    vector<pair<string, double>> top3imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 3);
    // Lấy top 5 ảnh
    vector<pair<string, double>> top5imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 5);
    // Lấy top 11 ảnh
    vector<pair<string, double>> top11imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 11);

    // Làm thành một vector
    vector<vector<pair<string, double>>> top3_5_11_21 = { top3imagePaths, top5imagePaths, top11imagePaths, top21imagePaths };

    // 4. Với mỗi vector cặp đó thực hiện vòng lặp vector cặp lấy url ra và so sánh
    int i = 0; // biến đếm tạm
    for (const auto& vP : top3_5_11_21) {
        int n = numberOfImagesList[i++];
        int relevantCount = 0;
        float sumPrecision = 0.0;

        // For auto pair in vector pair
        for (int k = 0; k < vP.size(); ++k) {
            const auto& pair = vP[k];
            const string& url = pair.first;

            // Tìm đường dẫn URL tương ứng trong records
            auto it = std::find_if(this->records.begin(), this->records.end(), [&](const myRecord& r) {
                return r.URL == url;
                });

            if (it != this->records.end() && it->Label == queryLabel) {
                // Nếu label query giống với label trong record, thực hiện +relevantCount (groundtruth)
                // Giả sử tập top3 image = [A, B, A] và label = A
                // => 1/1 + 0 + 2/3 = 5/3
                // Tương tự với tập vector top5, 11...
                relevantCount++;
                sumPrecision += static_cast<float>(relevantCount) / (k + 1);
            }
        }

        // Tính AP và lưu vào vector APs
        // (5/3)/2 = 5/6 và lưu vào file
        float AP = (relevantCount == 0) ? 0.0f : sumPrecision / relevantCount;
        APs.push_back(AP);
    }

    string pictureName = getFileNameFromFilePath(queryImagePath);

    // 5. Ghi kết quả MAP vào file CSV (lưu ý append)
    outputFile << pictureName << "," << queryLabel << ",";
    for (float mapValue : APs) {
        outputFile << mapValue << ",";
    }
    outputFile << endl;

    return;
}

// Tổng hợp n = 3, 5, 11, 21 ghi lại MAP vào một file CSV
// Có các cột là: URL,N=3,N=5,N=11,N=21
void ORBFeature::computeAndWriteMAP(const string& queryImagePath, const Size& size, ofstream& outputFile) {
    // 1. Khởi tạo MAPList để lưu vào và list returnImages
    vector<int> numberOfImagesList = { 3, 5, 11, 21 };
    vector<float> APs;

    // 2. Lấy label theo queryImage trong csvFile
    string queryLabel;
    bool foundQueryLabel = false;

    // 2.1 Tìm kiếm queryLabel trong records
    for (const auto& record : this->records) {
        if (record.URL == queryImagePath) {
            queryLabel = record.Label;
            foundQueryLabel = true;
            break;
        }
    }

    // 2.2 Không tìm thấy label trong file CSV
    if (!foundQueryLabel) {
        cout << "Query label not found in CSV file." << endl;
        return;
    }

    // 3. Thực hiện truy vấn 1 lần duy nhất với 21 tấm ảnh vì lúc nào cũng trả ra top 21 tấm
    vector<pair<string, double>> top21imagePaths = this->searchLocal(queryImagePath, 22, size);
    // Xóa ảnh đầu tiên (vì ảnh đầu tiên luôn là ảnh query)
    if (!top21imagePaths.empty() && top21imagePaths.front().first == queryImagePath) {
        top21imagePaths.erase(top21imagePaths.begin());
    }
    // Lấy top 3 ảnh
    vector<pair<string, double>> top3imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 3);
    // Lấy top 5 ảnh
    vector<pair<string, double>> top5imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 5);
    // Lấy top 11 ảnh
    vector<pair<string, double>> top11imagePaths(top21imagePaths.begin(), top21imagePaths.begin() + 11);

    // Làm thành một vector
    vector<vector<pair<string, double>>> top3_5_11_21 = { top3imagePaths, top5imagePaths, top11imagePaths, top21imagePaths };

    // 4. Với mỗi vector cặp đó thực hiện vòng lặp vector cặp lấy url ra và so sánh
    int i = 0; // biến đếm tạm
    for (const auto& vP : top3_5_11_21) {
        int n = numberOfImagesList[i++];
        int relevantCount = 0;
        float sumPrecision = 0.0;

        // For auto pair in vector pair
        for (int k = 0; k < vP.size(); ++k) {
            const auto& pair = vP[k];
            const string& url = pair.first;

            // Tìm đường dẫn URL tương ứng trong records
            auto it = std::find_if(this->records.begin(), this->records.end(), [&](const myRecord& r) {
                return r.URL == url;
                });

            if (it != this->records.end() && it->Label == queryLabel) {
                // Nếu label query giống với label trong record, thực hiện +relevantCount (groundtruth)
                // Giả sử tập top3 image = [A, B, A] và label = A
                // => 1/1 + 0 + 2/3 = 5/3
                // Tương tự với tập vector top5, 11...
                relevantCount++;
                sumPrecision += static_cast<float>(relevantCount) / (k + 1);
            }
        }

        // Tính AP và lưu vào vector APs
        // (5/3)/2 = 5/6 và lưu vào file
        float AP = (relevantCount == 0) ? 0.0f : sumPrecision / relevantCount;
        APs.push_back(AP);
    }

    string pictureName = getFileNameFromFilePath(queryImagePath);

    // 4. Ghi kết quả MAP vào file CSV (lưu ý append)
    outputFile << pictureName << "," << queryLabel << ",";
    for (float mapValue : APs) {
        outputFile << mapValue << ",";
    }
    outputFile << endl;

    return;
}

void SIFTFeature::processWriteMap(const string& MAPcsvPath, const Size& size, int maxRows) {
    // Mở file
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        return;
    }
    outputFile << "Picture Name,Label,N=3,N=5,N=11,N=21\n"; //ghi header
    // Thực hiện ghi tiếp các dòng qua vòng lặp
    // Giới hạn số lần ghi vào file
    int count = 1;
    for (const auto& r : this->records) {
        auto begin = high_resolution_clock::now();
        this->computeAndWriteMAP(r.URL, size, outputFile);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        cout << count << " - Iter duration: " << duration.count() << "ms\n";
        if (count >= maxRows) {
            cout << "Rows Limit reached, stopping...\n";
            break;
        }
        count++;
    }
}

void ORBFeature::processWriteMap(const string& MAPcsvPath, const Size& size, int maxRows) {
    // Mở file
    ofstream outputFile(MAPcsvPath, ios::app);
    if (!outputFile.good()) {
        cout << "Could not open file " << MAPcsvPath << endl;
        return;
    }
    outputFile << "Picture Name,Label,N=3,N=5,N=11,N=21\n"; //ghi header
    // Thực hiện ghi tiếp các dòng qua vòng lặp
    // Giới hạn số lần ghi vào file
    int count = 1;
    for (const auto& r : this->records) {
        auto begin = high_resolution_clock::now();
        this->computeAndWriteMAP(r.URL, size, outputFile);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - begin);
        cout << count << " - Iter duration: " << duration.count() << "ms\n";
        if (count >= maxRows) {
            cout << "Rows Limit reached, stopping...\n";
            break;
        }
        count++;
    }
}
