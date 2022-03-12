print("Hello")
#visual studioから

#tensorflow動作確認
import tensorflow as tf

#旧バージョン
#s = tf.Session()
#print(s.run(hello))

#新バージョン
hello = tf.constant('Hello World!')
tf.print(hello)