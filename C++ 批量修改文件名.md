# C++ 批量修改文件名

## 前言

在网上下一些学习资料，可是每个文件后带有一些其他无关的文字，形式如，某某某【某某某】.mp4，其中【】及其内容皆为无关内容，本文代码程序用于批量删除每个MP4文件后的【某某某】。

## 注意

文件名即为中文字符，不同于英文格式，所以以下代码中在需要的时候都使用了宽字符处理。

## 代码

```cpp
#include <iostream>
#include <string>
#include <vector>
#include <io.h>
int main()
{
	//使用宽字节流对象，绑定为中文
	locale china("chs");//use china character
	wcin.imbue(china);//use locale object
	wcout.imbue(china);

	wstring dirpath = L"E:\\test\\"; //注意宽字符或宽字符串在初始化时要加前缀L

	_wfinddata_t file;	//使用宽字节的_wfinddata_t对象而非_finddata_t
	long lf;	//是否遍历完毕的标志位

	wchar_t suffixs[] = L"*.mp4";   //要寻找的文件类型后缀，也统一使用宽字符串
	vector<wstring> fileNameList;   //文件夹下该类型文件的名字向量表
	wchar_t* p;
	int psize = dirpath.size() + 6;	//后面要把后缀加上，为了防止数组越界需要多开一点空间，6个正好
	p = new wchar_t[psize];
	wcscpy(p, dirpath.c_str());

	//获取文件名,存入向量表
	if ((lf = _wfindfirst(wcscat(p, suffixs), &file)) == -1l)
	{
		cout << "文件没有找到!\n";
	} else
	{
		cout << "\n文件列表:\n";
		do {
			//wcout << file.name << endl;
			wstring str(file.name);
			fileNameList.push_back(str);
			wcout << str << endl;
		} while (_wfindnext(lf, &file) == 0);
	}
	_findclose(lf);	//使用完毕后要关闭文件
	delete[] p;

	//遍历文件名向量表，并进行修改
	cout << "\n开始修改文件名：" << endl;
	for (vector<wstring>::iterator iter = fileNameList.begin(); iter != fileNameList.end(); ++iter)
	{
		wstring oldName = dirpath + *iter;	//记得加上绝对路径
		auto pos = iter->find(L"【");
		wstring newName = dirpath + iter->substr(0, pos);
		newName += L".mp4";

		wcout << "oldName:" << oldName << endl;
		wcout << "newName:" << newName << endl;

		wcout << "oldName size = " << oldName.size() << endl;
		wcout << "newName size = " << newName.size() << endl;
		int ret = _wrename(oldName.c_str(), newName.c_str());
		if (ret != 0)
			perror("rename error!");
		cout << endl;
	}
	system("pause");
	return 0;
}
```

读者可以根据自身需求修改代码。![img](https://dl4.weshineapp.com/gif/20220111/70e55b8d5163833dceb9ec9267092762.gif?f=micro_)

## 温馨提示

最好在使用程序前备用原资料，避免出现意外情况（比如我，在编写使用过程中忘记添加决定路径，导致原资料改名后跑到了项目路径下😑）。

![img](https://dl4.weshineapp.com/gif/20170814/b8184b3cc27a999410872f543388cdf6.gif?f=micro_)

## 参考

https://blog.csdn.net/Dr_Myst/article/details/81463450