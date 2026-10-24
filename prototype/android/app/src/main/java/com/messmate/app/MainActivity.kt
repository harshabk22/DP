package com.messmate.app

import android.annotation.SuppressLint
import android.app.Activity
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.View
import android.webkit.*
import android.widget.FrameLayout
import android.widget.ProgressBar
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.NotificationCompat
import androidx.core.content.FileProvider
import java.io.File

class MainActivity : ComponentActivity() {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar
    private var filePathCallback: ValueCallback<Array<Uri>>? = null
    private var cameraImageUri: Uri? = null
    private var cameraPhotoFile: File? = null

    private val prefs by lazy {
        getSharedPreferences("messmate_prefs", Context.MODE_PRIVATE)
    }

    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* Permission result handled */ }

    private val fileChooserLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (filePathCallback != null) {
            val uriResult: Array<Uri>? = if (result.resultCode == Activity.RESULT_OK) {
                val data = result.data
                if (data?.data != null) {
                    arrayOf(data.data!!)
                } else if (data?.clipData != null) {
                    val count = data.clipData!!.itemCount
                    Array(count) { i -> data.clipData!!.getItemAt(i).uri }
                } else if (cameraPhotoFile?.exists() == true && cameraPhotoFile!!.length() > 0 && cameraImageUri != null) {
                    arrayOf(cameraImageUri!!)
                } else null
            } else null

            filePathCallback?.onReceiveValue(uriResult)
            filePathCallback = null
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        cameraImageUri?.let { outState.putString("saved_camera_uri", it.toString()) }
        cameraPhotoFile?.let { outState.putString("saved_photo_path", it.absolutePath) }
    }

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        savedInstanceState?.let { bundle ->
            bundle.getString("saved_camera_uri")?.let { cameraImageUri = Uri.parse(it) }
            bundle.getString("saved_photo_path")?.let { cameraPhotoFile = File(it) }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
                notificationPermissionLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
            }
        }

        createNotificationChannel()

        val rootLayout = FrameLayout(this).apply {
            setBackgroundColor(android.graphics.Color.parseColor("#111110"))
        }

        webView = WebView(this).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(android.graphics.Color.parseColor("#111110"))
        }

        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                10
            )
            max = 100
            visibility = View.GONE
        }

        rootLayout.addView(webView)
        rootLayout.addView(progressBar)
        setContentView(rootLayout)

        configureWebView()

        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) {
                    webView.goBack()
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        loadMessMate()
        handleNotificationIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleNotificationIntent(intent)
    }

    private fun handleNotificationIntent(intent: Intent?) {
        val mealId = intent?.getStringExtra("meal_id")
        if (!mealId.isNullOrBlank()) {
            webView.post {
                webView.evaluateJavascript("window.loadMealById && window.loadMealById('$mealId');", null)
            }
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        val settings = webView.settings
        settings.javaScriptEnabled = true
        settings.domStorageEnabled = true
        settings.databaseEnabled = true
        settings.allowFileAccess = true
        settings.allowContentAccess = true
        settings.mediaPlaybackRequiresUserGesture = false
        settings.loadWithOverviewMode = true
        settings.useWideViewPort = true
        settings.setSupportZoom(false)
        settings.cacheMode = WebSettings.LOAD_DEFAULT
        settings.userAgentString = settings.userAgentString + " MessMate-Android/2.0"

        WebView.setWebContentsDebuggingEnabled(true)

        // Native JavaScript bridge for notification scheduling and URL configuration
        webView.addJavascriptInterface(MessMateNativeBridge(this), "MessMateNative")

        webView.webViewClient = object : WebViewClient() {
            override fun onReceivedError(
                view: WebView?,
                request: WebResourceRequest?,
                error: WebResourceError?
            ) {
                super.onReceivedError(view, request, error)
                if (request?.isForMainFrame == true) {
                    val currentUrl = request.url.toString()
                    val savedUrl = prefs.getString("server_url", "http://10.5.138.91:8000/") ?: "http://10.5.138.91:8000/"

                    // If failing on loopback, try Wi-Fi LAN failover once
                    if ((currentUrl.contains("localhost") || currentUrl.contains("127.0.0.1")) && !currentUrl.contains("10.5.138.91")) {
                        view?.postDelayed({
                            view.loadUrl("http://10.5.138.91:8000/")
                        }, 500)
                        return
                    }

                    // Fallback to interactive setup screen with custom IP input
                    val fallbackHtml = """
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <style>
                                body { background: #111110; color: #faf9f5; font-family: -apple-system, sans-serif; padding: 24px; text-align: center; }
                                h2 { color: #d97757; font-size: 20px; margin-top: 40px; }
                                p { color: #b0aea5; font-size: 14px; line-height: 1.5; }
                                input { width: 100%; box-sizing: border-box; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a19; color: #fff; margin: 16px 0; font-size: 14px; text-align: center; }
                                button { width: 100%; padding: 12px; border-radius: 8px; border: none; background: #d97757; color: #fff; font-weight: bold; font-size: 14px; margin-bottom: 12px; cursor: pointer; }
                                .secondary { background: #222220; color: #38bdf8; border: 1px solid #333; }
                            </style>
                        </head>
                        <body>
                            <h2>MessMate Offline / Connecting</h2>
                            <p>Cannot reach backend at:<br><code style="color:#d97757;">$currentUrl</code></p>
                            <p>Enter your Mac / PC server URL:</p>
                            <input id="serverIp" type="text" value="$savedUrl" placeholder="http://10.5.138.91:8000/" />
                            <button onclick="window.MessMateNative.setServerUrl(document.getElementById('serverIp').value)">Connect & Save</button>
                            <button class="secondary" onclick="window.MessMateNative.setServerUrl('http://127.0.0.1:8000/')">Retry via USB (127.0.0.1)</button>
                            <button class="secondary" onclick="window.location.reload()">Reload Page</button>
                        </body>
                        </html>
                    """.trimIndent()
                    view?.loadDataWithBaseURL(null, fallbackHtml, "text/html", "UTF-8", null)
                }
            }

            override fun onRenderProcessGone(view: WebView?, detail: RenderProcessGoneDetail?): Boolean {
                webView.destroy()
                recreate()
                return true
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView?, newProgress: Int) {
                if (newProgress < 100) {
                    progressBar.visibility = View.VISIBLE
                    progressBar.progress = newProgress
                } else {
                    progressBar.visibility = View.GONE
                }
            }

            override fun onShowFileChooser(
                webView: WebView?,
                filePathCallback: ValueCallback<Array<Uri>>?,
                fileChooserParams: FileChooserParams?
            ): Boolean {
                this@MainActivity.filePathCallback?.onReceiveValue(null)
                this@MainActivity.filePathCallback = filePathCallback

                return try {
                    // Create temporary photo file for camera capture
                    val photoFile = File.createTempFile(
                        "messmate_snap_${System.currentTimeMillis()}_",
                        ".jpg",
                        cacheDir
                    )
                    cameraPhotoFile = photoFile
                    val photoUri = FileProvider.getUriForFile(
                        this@MainActivity,
                        "${packageName}.fileprovider",
                        photoFile
                    )
                    cameraImageUri = photoUri

                    // 1. Direct camera capture intent with explicit ClipData and permissions
                    val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
                        putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
                        clipData = android.content.ClipData.newUri(contentResolver, "Plate Photo", photoUri)
                        addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }

                    // Explicitly grant URI permission to camera packages
                    val resInfoList = packageManager.queryIntentActivities(
                        takePictureIntent,
                        PackageManager.MATCH_DEFAULT_ONLY
                    )
                    for (resolveInfo in resInfoList) {
                        grantUriPermission(
                            resolveInfo.activityInfo.packageName,
                            photoUri,
                            Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION
                        )
                    }

                    // 2. Pure Gallery / Document picker intent
                    val galleryIntent = Intent(Intent.ACTION_GET_CONTENT).apply {
                        type = "image/*"
                        addCategory(Intent.CATEGORY_OPENABLE)
                    }

                    // 3. Combined Chooser
                    val chooserIntent = Intent.createChooser(galleryIntent, "Capture Plate or Select from Gallery").apply {
                        putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(takePictureIntent))
                    }

                    fileChooserLauncher.launch(chooserIntent)
                    true
                } catch (e: Exception) {
                    try {
                        val fallbackIntent = Intent(Intent.ACTION_GET_CONTENT).apply {
                            type = "image/*"
                        }
                        fileChooserLauncher.launch(fallbackIntent)
                        true
                    } catch (ex: Exception) {
                        this@MainActivity.filePathCallback?.onReceiveValue(null)
                        this@MainActivity.filePathCallback = null
                        false
                    }
                }
            }
        }
    }

    private fun loadMessMate() {
        val targetUrl = prefs.getString("server_url", "http://127.0.0.1:8000/") ?: "http://127.0.0.1:8000/"
        webView.loadUrl(targetUrl)
    }

    fun saveAndReloadUrl(newUrl: String) {
        val cleanUrl = if (!newUrl.endsWith("/")) "$newUrl/" else newUrl
        prefs.edit().putString("server_url", cleanUrl).apply()
        runOnUiThread {
            webView.loadUrl(cleanUrl)
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "MessMate Meal Clarifications",
                NotificationManager.IMPORTANCE_DEFAULT
            ).apply {
                description = "Ambient T+20 min portion and tableware check prompts (Invariant I4)"
                enableVibration(true)
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    fun scheduleDelayedNotification(mealId: String, prompt: String, optionsJson: String, delaySeconds: Long) {
        // Enforce Invariant I4: Ambient delay T+20 min (or tested interval)
        Handler(Looper.getMainLooper()).postDelayed({
            showClarificationNotification(mealId, prompt)
        }, delaySeconds * 1000L)
    }

    private fun showClarificationNotification(mealId: String, prompt: String) {
        val intent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("meal_id", mealId)
        }
        val pendingIntent = PendingIntent.getActivity(
            this,
            mealId.hashCode(),
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("MessMate • Quick Dining Check")
            .setContentText(prompt)
            .setStyle(NotificationCompat.BigTextStyle().bigText(prompt))
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .build()

        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(mealId.hashCode(), notification)
    }

    override fun onDestroy() {
        webView.destroy()
        super.onDestroy()
    }

    companion object {
        const val CHANNEL_ID = "messmate_clarifications"
    }

    class MessMateNativeBridge(private val activity: MainActivity) {
        @JavascriptInterface
        fun scheduleDelayedNotification(mealId: String, prompt: String, optionsJson: String, delaySeconds: Long) {
            activity.scheduleDelayedNotification(mealId, prompt, optionsJson, delaySeconds)
        }

        @JavascriptInterface
        fun setServerUrl(url: String) {
            activity.saveAndReloadUrl(url)
        }

        @JavascriptInterface
        fun isNativeAndroid(): Boolean = true
    }
}
