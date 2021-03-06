# web_ml

cryptocurrency_botをWeb上で動作させるためのプログラムです。

アカウント登録とログイン、ログアウトができます。

実行ボタンを押すと、停止ボタンを押すまで自動売買を続けます。

## 作成理由

自動売買プログラムをWeb上で管理したい

友人も使用できるようにしたい

などの理由から作成しようと考えました。

## 使用技術

Python 3.8.0

Django 3.0

sqlite 3.0

Celery 4.3.0

## アピールポイント

### Django

Djangoは、pythonで記述するWebアプリケーションのフレームワークです。

このフレームワークはフルスタックで、Webシステム開発に必要な機能がほぼ全て用意されています。

そのため学習コストが高く、扱えるようになるまで図書館で解説書を借りて学習しました。

### アカウント機能と暗号化

APIを用いて売買するためにはAPIキーとSECRETキーが必要です。

しかしAPIキーはランダムな数列のため覚えづらく、毎回要求するとアクセシビリティが悪くなります。

そのためアカウント方式にすることで、覚えやすいユーザー名とパスワードに、APIキーとSECRETキーを紐づけました。

APIキーを平文のままデータベースに保存するとセキュリティの観点から危険なので、パスワードを変換した鍵を用いてAESで暗号化したAPIキーを保存しています。

### 非同期処理

取引開始ボタンを押した後は\cryptocurrency_bot\ml_bot.py をwebアプリケーションとは別にループさせる必要があります。

そのためジョブキュー処理フレームワークのCeleryとブローカーのRabbitMQを使用して非同期処理を実装し、取引部分のプログラムを非同期化しました。
