#include "utils2.h"

bool contains(const vector<string>& vec, const string& str) {
    return find(vec.begin(), vec.end(), str) != vec.end();
}

// Các tham số mặc định cho chương trình
const vector<string> globalType = { "histogram" };
const vector<string> localType = { "sift", "orb" };

int main(int argc, char* argv[])
{
    //cout << argv[0] << endl; // file name
    //cout << argv[1] << endl; // argument 1:
    //cout << argv[2] << endl; // argument 2:
    //cout << argv[3] << endl; // argument 3:
    //cout << argv[4] << endl; // argument 4:

    // Siêu tham số
    // Các kích thước Size theo dataset 
    // [720 x 1280] -> [36 x 64] [72 x 128] [108 x 192] [144 x 256] [180 x 320] [216 x 384] [288 x 512]
    Size s144_256(144, 256);
    Size sizeUsing = s144_256;
    int clusterCount = 100;

    // Usage: <application name> <path/to/config/file.yml>
    string pathToConfigFile = argv[1];
    if (argc < 2) {
        cout << "Not enough arguments!" << endl;
        cout << "Usage: <file name> <path/to/your/config>" << endl;
        return -1;
    }
    //string pathToConfigFile = "<path/to/config/file.yml>";

    // Đọc file config
    FileStorage fs(pathToConfigFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cout << "Failed to open file config - path: " << pathToConfigFile << endl;
        return -1;
    }

    // Truy vấn theo loại nào (histogram, orb hay sift)
    string queryType;
    if (fs["queryType"].empty()) {
        cout << "queryType not found" << endl;
        return -1;
    }
    fs["queryType"] >> queryType;
    queryType = toLowerCase(queryType);

    // Đường dẫn tới CSDL ảnh
    string imageFolderPath;
    if (fs["imageFolderPath"].empty()) {
        cout << "imageFolderPath not found" << endl;
        return -1;
    }
    fs["imageFolderPath"] >> imageFolderPath;

    // Đường dẫn tới CSDL đặc trưng
    string descriptorsFilePath;
    if (fs["descriptorsFilePath"].empty()) {
        cout << "descriptorsFilePath not found" << endl;
        return -1;
    }
    fs["descriptorsFilePath"] >> descriptorsFilePath;

    // Đường dẫn tới file txt chứa URL khi lấy ảnh ra để tính descriptor(cần cho sift, orb)
    string URLFilePath;
    if (contains(localType, queryType)) {
        if (fs["URLFilePath"].empty()) {
            cout << "URLFilePath not found" << endl;
            return -1;
        }
        fs["URLFilePath"] >> URLFilePath;
    }

    // Đường dẫn đến tập dữ liệu csv gồm các cột name, label...
    string csvFilePath;
    if (fs["csvFilePath"].empty()) {
        cout << "csvFilePath not found" << endl;
        return -1;
    }
    fs["csvFilePath"] >> csvFilePath;

    // Có thực hiện kmeans cho truy vấn không
    bool usingKmeans = false;
    string centersFilePath;
    if (contains(localType, queryType)) {
        fs["usingKmeans"] >> usingKmeans;
        //cout << usingKmeans;
        if (usingKmeans) {
            if (fs["centersFilePath"].empty()) {
                cout << "centersFilePath not found" << endl;
                return -1;
            }
            fs["centersFilePath"] >> centersFilePath;
        }
    }

    // Giải phóng đối tượng fs
    fs.release();

    cout << "=== INPUT ===" << endl;
    cout << "-- image folder path: " << imageFolderPath << endl;
    cout << "-- image descriptors file path: " << descriptorsFilePath << endl;
    cout << "-- URL file path: " << URLFilePath << endl;
    cout << "-- dataset (csv) file path: " << csvFilePath << endl << endl;

    // Thực hiện quá trình tạo, đọc, truy vấn, hiển thị ảnh theo queryType
    if (contains(globalType, queryType)) {
        cout << "You are currently using " << queryType << endl;
        processGlobalFeature(queryType, imageFolderPath, descriptorsFilePath, csvFilePath);
    }
    else if (contains(localType, queryType)) {
        if (usingKmeans) {
            cout << "You are currently using " << queryType << " with Kmeans" << endl;
            cout << "=== ADDITIONAL INPUT ===" << endl;
            cout << "-- centers file path (kmeans's centers): " << centersFilePath << endl;
            cout << "-- size (fixed): 144x256" << endl;
            cout << "-- cluster number (codeword) (also fixed): 100" << endl;
            processLocalFeatureKmeans(queryType, imageFolderPath, descriptorsFilePath, URLFilePath, csvFilePath, centersFilePath, sizeUsing, clusterCount);
        }      
        else {
            cout << "You are currently using " << queryType << " without Kmeans" << endl;
            cout << "=== ADDITIONAL INPUT ===" << endl;
            cout << "-- size (fixed): 144x256" << endl;
            processLocalFeature(queryType, imageFolderPath, descriptorsFilePath, URLFilePath, csvFilePath, sizeUsing);
        }
    }
    else {
        cout << "-- Input query type: " << queryType << endl;
        cout << "-- Query type not accepted, must be (sift, orb or histogram)" << endl;
    }
    
    return 0;
}
