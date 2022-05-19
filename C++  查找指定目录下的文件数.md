# C++  查找指定目录下的文件数

## 代码

```cpp
#include<iostream>
using namespace std;
#include <filesystem>
#include <string>
using namespace std::filesystem;

// 注意这个函数也会统计隐藏文件
int getFileNumber(string folderPath) {
    if (!exists(folderPath))		// 如果目录不存在
        return 0;
    path root(folderPath);
    //如果路径不是目录也不是文件 
    if (!is_directory(root) && is_regular_file(root)) {
        return 0;
    }
    // 如果只是单个文件，返回1
    if (is_regular_file(root)) {
        return 1;
    }
    // DFS
    stack<path> st;
    st.push(root);
    int files = 0;
    while (!st.empty()) {
        path folder = st.top();
        st.pop();
        directory_entry entry(folder);	//文件入口
        directory_iterator list(entry);	//文件入口容器
        for (auto& it : list) {
            if (is_regular_file(it)) {	//文件的话，文件数++
                ++files;
            }
            if (is_directory(it)) {	//目录的话，入栈
                st.push(it);
            }
        }
    }
    return files;
}

int main()
{
    // 使用时传入指定目录字符串即可
    // 示例：查找E盘下test目录下的文件数，注意字符串结尾需加"\\"表示该路径是个目录形式
    cout<<getFileNumber("E:\\test\\");
    return 0;
}
```

## C++ 17 filesystem

借此浅学一下C++17 filesystem的文件管理操作

### 头文件及其命名空间

```cpp
#include<filesystem>
using namespace std::filesystem;
```

### 常用类

-   path 类：说白了该类只是对字符串（路径）进行一些处理，这也是文件系统的基石。
-   directory_entry 类：功如其名，文件入口，这个类才真正接触文件。 
-   directory_iterator 类：获取文件系统目录中文件的迭代器容器，其元素为 directory_entry对象（可用于遍历目录）
-   file_status 类：用于获取和修改文件（或目录）的属性（需要了解C++11的强枚举类型（即枚举类））

### 使用流程

1.  需要有一个path对象为基础，如果需要修改路径，可以调用其成员函数进行修改（注意其实只是处理字符串）。
2.  需要获取文件信息需要通过path构造directory_entry，但需要path一定存在才能调用构造，所以需要实现调用exists(path .)函数确保目录存在才能构造directory_entry（注意文件入口中的exists无法判断）。
3.  若需遍历，则可以使用 directory_iterator，进行遍历

### 实例

```cpp
#include <iostream>
#include<filesystem>
using namespace std;
using namespace std::filesystem;
int main(){
	path str("C:\\Windows");
	if (!exists(str))		//必须先检测目录是否存在才能使用文件入口.
		return 1;
	directory_entry entry(str);		//文件入口
	if (entry.status().type() == file_type::directory)	//这里用了C++11的强枚举类型
		cout << "该路径是一个目录" << endl;
	directory_iterator list(str);	        //文件入口容器
	for (auto& it:list) 
		cout << it.path().filename()<< endl;	//通过文件入口（it）获取path对象，再得到path对象的文件名，将之输出
	system("pause");
	return 0;
}
```

### 常用函数

void copy(const path& from, const path& to) ：目录复制

path absolute(const path& pval, const path& base = current_path()) ：获取相对于base的绝对路径

bool create_directory(const path& pval) ：当目录不存在时创建目录

bool create_directories(const path& pval) ：形如/a/b/c这样的，如果都不存在，创建目录结构

bool exists(const path& pval) ：用于判断path是否存在

uintmax_t file_size(const path& pval) ：返回目录的大小

file_time_type last_write_time(const path& pval) ：返回目录最后修改日期的file_time_type对象

bool remove(const path& pval) ：删除目录

uintmax_t remove_all(const path& pval) ：递归删除目录下所有文件，返回被成功删除的文件个数

void rename(const path& from, const path& to) ：移动文件或者重命名


想了解更多信息的话，[点这里](https://docs.microsoft.com/zh-cn/cpp/standard-library/filesystem?view=msvc-170)

参考：https://blog.csdn.net/qq_40946921/article/details/91394589，https://docs.microsoft.com/zh-cn/cpp/standard-library/filesystem?view=msvc-170

