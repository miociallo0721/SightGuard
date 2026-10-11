package org.sightguard.app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import android.util.Log
import java.util.ArrayDeque
import kotlin.math.sqrt

/** Bounded raw sensor histories. No acceleration integration, position, or navigation pose. */
class ImuMonitor(context: Context) : SensorEventListener, AutoCloseable {
    private val manager=context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val thread=HandlerThread("SightGuard-IMU").apply { start() }
    private data class Gyro(val timeNs:Long,val speed:Float)
    private data class Orientation(val timeNs:Long,val pitch:Float,val roll:Float)
    private val gyros=ArrayDeque<Gyro>()
    private val orientations=ArrayDeque<Orientation>()
    @Volatile private var closed=false
    private var count=0
    private val types=listOf(Sensor.TYPE_ACCELEROMETER,Sensor.TYPE_GYROSCOPE,Sensor.TYPE_GRAVITY,Sensor.TYPE_ROTATION_VECTOR)
    val available:List<Int> = types.filter { manager.getDefaultSensor(it)!=null }
    init {
        available.forEach { type -> manager.registerListener(this,manager.getDefaultSensor(type),SensorManager.SENSOR_DELAY_GAME,Handler(thread.looper)) }
        Log.i("SG_IMU","registered types=$available clock=elapsedRealtimeNanos")
    }
    @Synchronized override fun onSensorChanged(event:SensorEvent) {
        if(closed) return
        count++
        if(event.accuracy==SensorManager.SENSOR_STATUS_UNRELIABLE || event.values.any { !it.isFinite() }) return
        when(event.sensor.type) {
            Sensor.TYPE_GYROSCOPE -> {
                gyros.addLast(Gyro(event.timestamp,sqrt(event.values.take(3).sumOf { (it*it).toDouble() }).toFloat()))
                while(gyros.size>512) gyros.removeFirst()
            }
            Sensor.TYPE_ROTATION_VECTOR -> {
                val matrix=FloatArray(9); val angles=FloatArray(3)
                SensorManager.getRotationMatrixFromVector(matrix,event.values); SensorManager.getOrientation(matrix,angles)
                orientations.addLast(Orientation(event.timestamp,Math.toDegrees(angles[1].toDouble()).toFloat(),Math.toDegrees(angles[2].toDouble()).toFloat()))
                while(orientations.size>512) orientations.removeFirst()
            }
        }
        if(count%200==1) Log.i("SG_IMU","samples=$count type=${event.sensor.type} timestampNs=${event.timestamp} receiveNs=${SystemClock.elapsedRealtimeNanos()} values=${event.values.take(3)} accuracy=${event.accuracy}")
    }
    @Synchronized fun evidence(key:SpatialFrameKey,processedNs:Long=SystemClock.elapsedRealtimeNanos()):MotionEvidence {
        val capture=key.captureTimestampNs
        if(key.timestampClock!="REALTIME" || capture==null || capture<=0 || processedNs<capture) return MotionRules.classify(null,null,false)
        val preceding=gyros.lastOrNull { it.timeNs<=capture }
        val window=gyros.filter { it.timeNs>=capture-100_000_000L && it.timeNs<=processedNs }
        val gap=preceding?.let { capture-it.timeNs }
        val orientation=orientations.lastOrNull { it.timeNs<=capture && capture-it.timeNs<=200_000_000L }
        val poseWindow=orientations.filter { it.timeNs>=capture && it.timeNs<=processedNs }
        fun angleDifference(a:Float,b:Float):Float = kotlin.math.abs(((a-b+540f)%360f)-180f)
        val delta=orientation?.let { first -> poseWindow.maxOfOrNull { maxOf(angleDifference(first.pitch,it.pitch),angleDifference(first.roll,it.roll)) } }
        val result=MotionRules.classify(window.maxOfOrNull { it.speed },gap,true,orientation?.pitch,orientation?.roll,orientationDeltaDegrees=delta)
        return result.copy(sampleCount=window.size)
    }
    @Synchronized fun currentDeviceMotion(nowNs:Long=SystemClock.elapsedRealtimeNanos()):MotionEvidence {
        val last=gyros.lastOrNull()
        val window=gyros.filter { nowNs-it.timeNs in 0..200_000_000L }
        return MotionRules.classify(window.maxOfOrNull { it.speed },last?.let { nowNs-it.timeNs },true)
            .copy(synchronized=false,reason="current device sensor window only; no source frame alignment",sampleCount=window.size)
    }
    @Synchronized fun diagnostic():String = "设备运动 ${currentDeviceMotion().quality} · samples $count · gyro ${gyros.lastOrNull()?.speed ?: "UNKNOWN"} rad/s · 帧同步另行检查"
    override fun onAccuracyChanged(sensor:Sensor?,accuracy:Int)=Unit
    @Synchronized override fun close() {
        if(closed) return
        closed=true; manager.unregisterListener(this); gyros.clear(); orientations.clear(); thread.quitSafely()
        Log.i("SG_IMU","released samples=$count")
    }
}
