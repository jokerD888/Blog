# Ubuntu20.0下安装MySQL8.0

## 前言

时间：2022.9.20

我的linux版本为<code>20.04.1</code>，可通过<code>uname -a</code>查看。所安装的MySQL版本为<code>8.0</code> 安装完毕后，可通过<code>mysql -V</code> 查看。

目的为记录安装过程，以及其中遇到的一些问题。

如果先前安装过mysql，想要重新安装，可使用如下命令完全卸载清理mysql。

```bash
sudo apt purge mysql-*
sudo rm -rf /etc/mysql/ /var/lib/mysql
sudo apt autoremove
sudo apt autoclean
```

## 正文

###  下载MySQL APT安装配置包。

该方案使用的是MySQL官方的软件源，如果网络连通状况不佳，请参考 [Ubuntu 安装MySQL(国内镜像源)](https://blog.csdn.net/weixin_44129085/article/details/105403674) 使用国内镜像源，以提高安装时下载速度。

-   首先访问 https://dev.mysql.com/downloads/repo/apt/ 获取配置包下载地址
    ![](Ubuntu20.0%E4%B8%8B%E5%AE%89%E8%A3%85MySQL8.0.assets/mysql(1).png)
-   复制下载链接
    ![](Ubuntu20.0%E4%B8%8B%E5%AE%89%E8%A3%85MySQL8.0.assets/mysql(2).png)

​		目前复制到的链接为

```
https://dev.mysql.com/get/mysql-apt-config_0.8.23-1_all.deb
```

​		

-   进入Ubuntu系统，打开终端，并输入以下命令，进行下载MySQL APT配置包

    ```bash
    wget https://dev.mysql.com/get/mysql-apt-config_0.8.23-1_all.deb
    ```

-   进入主目录，可以看到已下载好的MySQL APT配置包

### 安装MySQL APT配置包

-   输入以下命令，进行安装：

    ```bash
    sudo https://dev.mysql.com/get/mysql-apt-config_0.8.23-1_all.deb
    ```

    安装过程中出现选择项，通过上下键选择OK继续安装即可。
    ![](Ubuntu20.0%E4%B8%8B%E5%AE%89%E8%A3%85MySQL8.0.assets/mysql(3).png)

    若出现让你选择系统的页面，按esc退出，再进入即可。

-   安装完成后，最后一行会出现OK

### 安装MySQL Server

-   更新**APT**软件源：

    ```bash
    sudo apt-get update
    ```

-   安装MySQL Server

    ```bash
    sudo apt-get install mysql-server
    ```

-   输入 `y` 继续执行，弹出MySQL 8安装对话框，按回车键确定，进入设置root密码的对话框。若没有出现，说明其自动安装完成了，若mysql -uroot -p，可能会出现如下错误。我个人出现这种情况。

    ```bash
    ERROR 1698 (28000): Access denied for user 'root'@'localhost'
    ```

    那么参照[这里](https://stackoverflow.com/questions/39281594/error-1698-28000-access-denied-for-user-rootlocalhost)既可解决。解决后

-   接下来，按照GUI界面设置密码并重复和确定。

-   按照完成。

-   MySQL 8安装好之后，会创建如下目录

    ```bash
    数据库目录：/var/lib/mysql/。
    配置文件：/usr/share/mysql-8.0（命令及配置文件），/etc/mysql（如my.cnf）。
    相关命令：/usr/bin（mysqladmin、mysqldump等命令）和/usr/sbin。
    启动脚本：/etc/init.d/mysql（启动脚本文件mysql的目录）。
    ```

### 启动MySQL服务

-   通过以上的APT方式安装好之后，所有的服务、环境变量都会启动和配置好，无须手动配置。

-   服务器启动后端口查询

    ```bash
    sudo netstat -anp | grep mysql
    ```

-   服务管理

    ```bash
    查看服务状态
    sudo service mysql status
    停止服务
    sudo service mysql stop
    启动服务
    sudo service mysql start
    重启服务
    sudo service mysql restart
    ```

### 登录MySQL数据库

```bash
mysql -u root -p
```

然后输入你刚在GUI界面输入的密码，即可登录。

## 其他问题

安装后若没有mysql.h文件，即mysql.h在ubuntu下默认安装在/user/include/mysql/mysql.h，若没有没有mysql目录或目录下没有文件，可用一下命令安装`mysql`的相关链接库。

```bash
sudo apt-get install libmysqlclient-dev
```

为用户设置密码

```bash
mysql> ALTER USER 'root'@'localhost' IDENTIFIED WITH caching_sha2_password BY '123456';
```

