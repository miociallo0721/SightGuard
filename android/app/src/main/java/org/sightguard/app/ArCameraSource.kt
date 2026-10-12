package org.sightguard.app

import android.app.Activity
import android.graphics.Bitmap
import android.graphics.Matrix
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.Image
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.SystemClock
import android.util.Log
import com.google.ar.core.*
import com.google.ar.core.exceptions.NotYetAvailableException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/** Owns the camera exclusively. One CPU frame in flight; GL preview never waits for inference. */
class ArCameraSource(
    private val activity: Activity,
    private val onFrame: (Bitmap, ArGeometry, () -> Unit) -> Unit,
    private val onFailure: (String) -> Unit,
) : GLSurfaceView.Renderer, AutoCloseable {
    val session = Session(activity)
    val view = GLSurfaceView(activity)
    private val busy = AtomicBoolean(false)
    @Volatile private var closed = false
    private var texture = 0
    private var program = 0
    private var lastTimestamp = 0L
    private var lastImageTimestamp = 0L
    private var previewFrames = 0
    private var width = 1
    private var height = 1
    private val cameraSize: android.util.Size
    private val textureSize: android.util.Size
    private val sensorOrientation: Int
    private val realtimeCameraClock: Boolean
    @Volatile var previewMapping: ArPreviewMapping? = null
        private set
    private val positions = buffer(floatArrayOf(-1f,-1f,1f,-1f,-1f,1f,1f,1f))
    private val uv = buffer(FloatArray(8))
    init {
        try {
            check(session.isDepthModeSupported(Config.DepthMode.AUTOMATIC)) { "DEPTH_UNSUPPORTED" }
            val candidates=session.getSupportedCameraConfigs(CameraConfigFilter(session))
            candidates.filter {it.imageSize.width*it.imageSize.height<=1280*720}
                .minByOrNull {it.imageSize.width*it.imageSize.height}?.let {session.cameraConfig=it}
            cameraSize=session.cameraConfig.imageSize
            textureSize=session.cameraConfig.textureSize
            val characteristics=(activity.getSystemService(Activity.CAMERA_SERVICE) as CameraManager).getCameraCharacteristics(session.cameraConfig.cameraId)
            sensorOrientation=characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            realtimeCameraClock=characteristics.get(CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE)==CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
            session.configure(Config(session).apply {
                depthMode=Config.DepthMode.AUTOMATIC
                focusMode=Config.FocusMode.AUTO
                updateMode=Config.UpdateMode.LATEST_CAMERA_IMAGE
                planeFindingMode=Config.PlaneFindingMode.DISABLED
                lightEstimationMode=Config.LightEstimationMode.DISABLED
            })
            view.setEGLContextClientVersion(2)
            view.preserveEGLContextOnPause=true
            view.setRenderer(this)
            session.resume()
            Log.i("SG_AR","session resumed depth=true cpu=$cameraSize sensorOrientation=$sensorOrientation realtimeClock=$realtimeCameraClock elapsedNs=${SystemClock.elapsedRealtimeNanos()} monotonicNs=${System.nanoTime()}")
        } catch(error: Exception) { session.close(); throw error }
    }
    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val textures=IntArray(1);GLES20.glGenTextures(1,textures,0);texture=textures[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,texture)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE)
        fun shader(type: Int, source: String): Int {
            val id=GLES20.glCreateShader(type);GLES20.glShaderSource(id,source);GLES20.glCompileShader(id)
            val ok=IntArray(1);GLES20.glGetShaderiv(id,GLES20.GL_COMPILE_STATUS,ok,0)
            check(ok[0]!=0) {GLES20.glGetShaderInfoLog(id)};return id
        }
        val vertex=shader(GLES20.GL_VERTEX_SHADER,"attribute vec2 p; attribute vec2 t; varying vec2 v; void main(){gl_Position=vec4(p,0.,1.);v=t;}")
        val fragment=shader(GLES20.GL_FRAGMENT_SHADER,"#extension GL_OES_EGL_image_external : require\nprecision mediump float; uniform samplerExternalOES s; varying vec2 v; void main(){gl_FragColor=texture2D(s,v);}")
        program=GLES20.glCreateProgram();GLES20.glAttachShader(program,vertex);GLES20.glAttachShader(program,fragment)
        GLES20.glLinkProgram(program);GLES20.glDeleteShader(vertex);GLES20.glDeleteShader(fragment)
        session.setCameraTextureName(texture)
    }
    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {width=w;height=h}
    @Suppress("DEPRECATION")
    override fun onDrawFrame(gl: GL10?) {
        if(closed) return
        try {
            val displayRotation=activity.windowManager.defaultDisplay.rotation
            val rotation=(sensorOrientation-displayRotation*90+360)%360
            val uprightWidth=if(rotation%180==0) textureSize.width else textureSize.height
            val uprightHeight=if(rotation%180==0) textureSize.height else textureSize.width
            val scale=minOf(width.toFloat()/uprightWidth,height.toFloat()/uprightHeight)
            val vw=(uprightWidth*scale).toInt().coerceAtLeast(1);val vh=(uprightHeight*scale).toInt().coerceAtLeast(1)
            session.setDisplayGeometry(displayRotation,vw,vh)
            val frame=session.update()
            if(++previewFrames%300==1) Log.i("SG_AR","preview frameNs=${frame.timestamp} previousNs=$lastTimestamp busy=${busy.get()} tracking=${frame.camera.trackingState}")
            GLES20.glViewport(0,0,width,height);GLES20.glClearColor(0f,0f,0f,1f);GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
            if(frame.timestamp==0L) return
            // Official display-to-texture transform avoids sampling beyond the GPU crop.
            val transformed=FloatArray(8)
            frame.transformCoordinates2d(Coordinates2d.OPENGL_NORMALIZED_DEVICE_COORDINATES,floatArrayOf(-1f,-1f,1f,-1f,-1f,1f,1f,1f),Coordinates2d.TEXTURE_NORMALIZED,transformed)
            val imageBasis=floatArrayOf(0f,0f,1f,0f,0f,1f)
            for(i in 0..2) {val p=ArGeometryRules.unrotate(imageBasis[2*i],imageBasis[2*i+1],rotation);imageBasis[2*i]=p.first;imageBasis[2*i+1]=p.second}
            val viewBasis=FloatArray(6)
            frame.transformCoordinates2d(Coordinates2d.IMAGE_NORMALIZED,imageBasis,Coordinates2d.VIEW_NORMALIZED,viewBasis)
            previewMapping=ArPreviewMapping(viewBasis.toList(),vw.toFloat()/width,vh.toFloat()/height)
            uv.position(0);uv.put(transformed);uv.position(0)
            GLES20.glViewport((width-vw)/2,(height-vh)/2,vw,vh);GLES20.glUseProgram(program)
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,texture)
            GLES20.glUniform1i(GLES20.glGetUniformLocation(program,"s"),0)
            val p=GLES20.glGetAttribLocation(program,"p");val t=GLES20.glGetAttribLocation(program,"t")
            GLES20.glEnableVertexAttribArray(p);GLES20.glEnableVertexAttribArray(t)
            positions.position(0);GLES20.glVertexAttribPointer(p,2,GLES20.GL_FLOAT,false,0,positions)
            GLES20.glVertexAttribPointer(t,2,GLES20.GL_FLOAT,false,0,uv);GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4)
            GLES20.glDisableVertexAttribArray(p);GLES20.glDisableVertexAttribArray(t)
            if(frame.timestamp==lastTimestamp || !busy.compareAndSet(false,true)) return
            lastTimestamp=frame.timestamp
            var bitmap: Bitmap?=null
            try {
                val received=SystemClock.elapsedRealtime()
                frame.acquireCameraImage().use {image ->
                    // acquireCameraImage is the image for this Frame. Its timestamp is not
                    // specified to equal either ARCore timestamp or the Android capture timestamp.
                    // Keep API association and reject repeated images instead of inventing offsets.
                    if(image.timestamp<=lastImageTimestamp) {busy.set(false);return}
                    lastImageTimestamp=image.timestamp
                    bitmap=yuvBitmap(image,rotation)
                }
                val pose=frame.camera.pose
                val points=mutableListOf<ArDepthPoint>();var depthTimestamp=0L
                if(frame.camera.trackingState==TrackingState.TRACKING) try {
                    frame.acquireRawDepthImage16Bits().use {depth -> frame.acquireRawDepthConfidenceImage().use {confidence ->
                        depthTimestamp=depth.timestamp
                        if(confidence.width==depth.width && confidence.height==depth.height && confidence.timestamp==depth.timestamp) {
                            val dw=depth.width;val dh=depth.height
                            val coords=mutableListOf<Float>();val depths=mutableListOf<Float>();val confidences=mutableListOf<Int>()
                            val dp=depth.planes[0];val cp=confidence.planes[0]
                            val db=dp.buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN);val cb=cp.buffer.duplicate()
                            val dr=dp.rowStride;val ds=dp.pixelStride;val cr=cp.rowStride;val cs=cp.pixelStride
                            val db0=db.position();val cb0=cb.position()
                            for(y in 0 until dh) for(x in 0 until dw) {
                                val z=(db.getShort(db0+y*dr+x*ds).toInt() and 65535)/1000f
                                val c=cb.get(cb0+y*cr+x*cs).toInt() and 255
                                if(z>0 && c>=200) {coords.add((x+.5f)/dw);coords.add((y+.5f)/dh);depths.add(z);confidences.add(c)}
                            }
                            if(depths.isNotEmpty()) {
                                val imageCoords=FloatArray(coords.size)
                                frame.transformCoordinates2d(Coordinates2d.TEXTURE_NORMALIZED,coords.toFloatArray(),Coordinates2d.IMAGE_NORMALIZED,imageCoords)
                                val intrinsics=frame.camera.imageIntrinsics;val focal=intrinsics.focalLength;val center=intrinsics.principalPoint
                                // Retain sparse high-confidence pixels before imposing the point budget.
                                val sampleStep=maxOf(1,(depths.size+1599)/1600)
                                for(i in depths.indices step sampleStep) {val z=depths[i]
                                    val x=imageCoords[2*i];val y=imageCoords[2*i+1]
                                    if(x in 0f..1f && y in 0f..1f) {
                                        val normalized=ArGeometryRules.rotate(x,y,rotation)
                                        val world=pose.transformPoint(floatArrayOf((x*cameraSize.width-center[0])*z/focal[0],-(y*cameraSize.height-center[1])*z/focal[1],-z))
                                        points.add(ArDepthPoint(normalized.first,normalized.second,z,confidences[i]/255f,Vec3(world[0],world[1],world[2])))
                                    }
                                }
                            }
                        }
                    } }
                } catch(_: NotYetAvailableException) { /* Empty is unknown, never zero distance. */ }
                val tr=pose.translation;val forward=pose.rotateVector(floatArrayOf(0f,0f,-1f))
                val packet=ArGeometry(frame.timestamp,depthTimestamp,received,rotation,frame.camera.trackingState.name,
                    frame.camera.trackingFailureReason.name,Vec3(tr[0],tr[1],tr[2]),Vec3(forward[0],forward[1],forward[2]),points,
                    if(realtimeCameraClock) frame.androidCameraTimestamp/1_000_000 else received,frame.androidCameraTimestamp,realtimeCameraClock)
                val owned=bitmap!!;bitmap=null
                onFrame(owned,packet) {owned.recycle();busy.set(false)}
            } catch(_: NotYetAvailableException) {bitmap?.recycle();busy.set(false)}
              catch(error: Exception) {bitmap?.recycle();busy.set(false);throw error}
        } catch(error: Exception) {
            if(!closed) {closed=true;Log.e("SG_AR","camera failed",error);activity.runOnUiThread {onFailure(error.javaClass.simpleName+": "+error.message)}}
        }
    }
    override fun close() {
        closed=true;view.onPause();runCatching {session.pause()};session.close()
        Log.i("SG_AR","session released")
    }
    companion object {
        private fun buffer(values: FloatArray)=ByteBuffer.allocateDirect(values.size*4).order(ByteOrder.nativeOrder()).asFloatBuffer().apply {put(values);position(0)}
        private fun yuvBitmap(image: Image, rotation: Int): Bitmap {
            val planes=image.planes;val bytes=planes.map {p -> val b=p.buffer.duplicate();ByteArray(b.remaining()).also {b.get(it)}}
            val w=image.width;val h=image.height;val ys=planes[0].rowStride;val us=planes[1].rowStride;val vs=planes[2].rowStride
            val yp=planes[0].pixelStride;val up=planes[1].pixelStride;val vp=planes[2].pixelStride
            val yb=bytes[0];val ub=bytes[1];val vb=bytes[2];val pixels=IntArray(w*h)
            for(y in 0 until h) for(x in 0 until w) {
                val yy=((yb[y*ys+x*yp].toInt() and 255)-16).coerceAtLeast(0)*298
                val u=(ub[y/2*us+x/2*up].toInt() and 255)-128
                val v=(vb[y/2*vs+x/2*vp].toInt() and 255)-128
                pixels[y*w+x]=(-16777216) or (((yy+409*v+128) shr 8).coerceIn(0,255) shl 16) or
                    (((yy-100*u-208*v+128) shr 8).coerceIn(0,255) shl 8) or (((yy+516*u+128) shr 8).coerceIn(0,255))
            }
            val raw=Bitmap.createBitmap(pixels,w,h,Bitmap.Config.ARGB_8888)
            if(rotation==0) return raw
            return Bitmap.createBitmap(raw,0,0,raw.width,raw.height,Matrix().apply {postRotate(rotation.toFloat())},false).also {if(it!==raw) raw.recycle()}
        }
    }
}
