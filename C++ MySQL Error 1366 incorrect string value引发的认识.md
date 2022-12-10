# C++ MySQL Error 1366 incorrect string value引发的认识

## 前言

在使用MySQL C++ API编写程序时，由于用到了中文，导致出现了MySQL error 1366 incorrect string value 问题，但令我同样不解的是我用同样的语句在cmd下可以正常执行。MySQL1366 报错如下：
![](E:/images/1336error.png)

虽然可以很容易发现时字符集编码的问题，也确实是这个问题，这也是第一次编写程序遇到，虽然找问题解决方法很痛苦，但对字符集编码以及跨平台有了更深的了解。

## 正文

在编写代码初始化数据库，设置字符集时，我认识到了把MySQL的utf8并不是我们普遍认为的utf8,  实际意义上应该是utf8mb4。起初我以为是MySQL的字符集编码设置有问题，甚至到表，字段的字符集都检查了但都没有问题，而且谷度也基本也没搜到有用的信息，大都是认为是数据库的字符集有问题。在这个过程中，我利用配置文件my.ini把数据库的相关编码都设置为了utf8mb4，这也许是导致我之前可以在命令行执行，后来却执行不了的原因。下面是关于几个系统变量的说明。

| 系统变量                 | 描述                       |
| ------------------------ | -------------------------- |
| character_set_client     | 客户端来源数据使用的字符集 |
| character_set_connection | 连接层字符集               |
| character_set_database   | 当前选中数据库的默认字符集 |
| character_set_results    | 查询结果字符集             |
| character_set_server     | 默认的内部操作字符集       |

然后问题终于有了突破性的进展。在处理问题的途中，我也得知了cmd下使用的是GBK中文编码。随后我又猜想是执行的语句不是utf8（实际的utf8)格式的原因（与character_set_client不符），所以我使用`chcp 65001`使得cmd下字符集设置为utf8，经过两个命令行窗口进行对比，果然是这个原因，utf8这边正常执行语句，进行正常显示，而原GBK这边，往字段中插入中文就会报错，同时之前原有的中文记录也出现乱码。另外MySQL 也提供了Command Line Client -Unicode 的窗口，在这下面执行，也符合mysql 的utf8mb4，所以也正确执行和显示。

最后，通过上面的验证，我把问题最后锁定在了代码中SQL语句的字符集问题上，随后我又经过一顿搜索，得到了如下字符串std::string转换为utf8模式得函数。

```cpp
std::string StringToUTF8(const std::string& gbkData)
    {
        const char* GBK_LOCALE_NAME = "CHS";  //GBK在windows下的locale name(.936, CHS ), linux下的locale名可能是"zh_CN.GBK"

        std::wstring_convert<std::codecvt<wchar_t, char, mbstate_t>>
            conv(new std::codecvt<wchar_t, char, mbstate_t>(GBK_LOCALE_NAME));
        std::wstring wString = conv.from_bytes(gbkData);    // string => wstring

        std::wstring_convert<std::codecvt_utf8<wchar_t>> convert;
        std::string utf8str = convert.to_bytes(wString);     // wstring => utf-8

        return utf8str;
    }
```

所以最终的解决方法是，通过执行sql语句前多加一层字符集的转换，问题都到了解决。花费了一晚上时间，此时这篇文章到此时间为2022.12.09.23:55，OK，完事，上床睡觉。

## 补充

通过这个问题，我对字符集有了一些新的认识（可能并不恰当）。

我认为问题的主要原因是系统变量character_set_client与实际数据使用的字符集不匹配所导致。验证如下，在不改变命令行窗口的字符集的情况下（默认GBK)，通过改变character_set_client的值，对比执行是否成功，更加验证了我的猜想。

![](C++%20MySQL%20Error%201366%20incorrect%20string%20value%E5%BC%95%E5%8F%91%E7%9A%84%E8%AE%A4%E8%AF%86.assets/1336error1-16706529895712.png)

同样输出乱码则和系统变量character_set_results与实际显示数据的字符集不匹配有关。验证如下。

![](C++%20MySQL%20Error%201366%20incorrect%20string%20value%E5%BC%95%E5%8F%91%E7%9A%84%E8%AE%A4%E8%AF%86.assets/1336error2.png)

总之，输入要和character_set_client字符集匹配，显示要和character_set_results字符集匹配，即我们的命令行窗口字符集要和character_set_client以及character_set_client相同。

通俗来说，character_set_client ，这是用户告诉MySQL查询是用的什么字符集。character_set_connection ，MySQL接受到用户查询后，按照character_set_client将其转化为character_set_connection设定的字符集。character_set_results ， MySQL将存储的数据转换成character_set_results中设定的字符集发送给用户。set names xxx等同于同时设置这三个。

此三处的字符设定很大程度上会解决乱码问题，那么着三个设定具体有什么作用呢？

character_set_client指定的是Sql语句的编码，如果设置为 binary，mysql就当二进制来处理，character_set_connection指定了mysql 用来运行sql语句的时候使用的编码，也就是说，程序发送给MySQL 的SQL语句,会首先被MySQL从character_set_client指定的编码转换到character_set_connection指定的编码，如果character_set_clien指定的是binary，则MySQL就会把SQL语句按照character_set_connection指定的编码解释执行.
		当执行SQL语句的过程中，比如向数据库中插入字段的时候，字段也有编码设置，如果字段的编码设置和character_set_connection指定的不同，则MySQL 会把插入的数据转换成字段设定的编码。SQL语句中的条件判断和SQL插入语句的执行过程类似.
		当SQL执行完毕像客户端返回数据的时候，会把数据从字段指定的编码转换为character_set_results指定的编码，如果character_set_results=NULL 则不做任何转换动作，（注意这里设置为NULL不等于没有设置，没有设置的时候MySQL会继承全局设置）,
工作中比较有用的就是利用MySQL进行转码、不同编码的数据库之间共用数据。
