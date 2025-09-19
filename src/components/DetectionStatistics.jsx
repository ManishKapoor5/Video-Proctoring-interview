import React, { useState, useEffect, useRef, useCallback } from 'react';
import { AlertTriangle, Eye, EyeOff, Users, Smartphone, FileText, Moon, Camera, CameraOff, Loader } from 'lucide-react';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const EAR_THRESHOLD = 0.2; // Eye Aspect Ratio threshold
const DROWSINESS_FRAMES = 15; // How many frames of closed eyes mean "drowsy"
const FOCUS_LOST_FRAMES = 35; // How many frames of "gaze away" mean lost focus

const DetectionStatistics = () => {
  const [stats, setStats] = useState({
    focusLost: 0,
    faceAbsent: 0,
    multipleFaces: 0,
    phoneDetected: 0,
    notesDetected: 0,
    drowsiness: 0
  });

  const [currentDetections, setCurrentDetections] = useState({
    focusLost: false,
    faceAbsent: false,
    multipleFaces: false,
    phoneDetected: false,
    notesDetected: false,
    drowsiness: false
  });

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState('');
  const [detectionResults, setDetectionResults] = useState([]);

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const detectionIntervalRef = useRef(null);
  const blazeFaceModelRef = useRef(null);
  const cocoSsdModelRef = useRef(null);

  // Focus and drowsiness counters
  const lastFacePosition = useRef({ x: 0, y: 0 });
  const focusLostFrames = useRef(0);
  const drowsyFrames = useRef(0);

  // Load models
  useEffect(() => {
    const loadModels = async () => {
      setLoadingStatus('Loading models...');
      blazeFaceModelRef.current = await blazeface.load();
      cocoSsdModelRef.current = await cocoSsd.load();
      setModelsLoaded(true);
      setLoadingStatus('');
    };
    loadModels();
  }, []);

  // Camera handling
  const startCamera = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: {facingMode:'user'} });
    streamRef.current = stream;
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }
    setCameraActive(true);
  };
  const stopCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(track => track.stop());
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
  };

  // Face detection
  const detectFacesTF = async (videoElement) => {
    if (!blazeFaceModelRef.current || !videoElement) return [];
    const predictions = await blazeFaceModelRef.current.estimateFaces(videoElement, false);
    return predictions.map(pred => ({
      x: pred.topLeft[0], y: pred.topLeft[1],
      width: pred.bottomRight[0] - pred.topLeft[0],
      height: pred.bottomRight[1] - pred.topLeft[1],
      centerX: (pred.topLeft[0] + pred.bottomRight[0]) / 2,
      centerY: (pred.topLeft[1] + pred.bottomRight[1]) / 2,
      probability: pred.probability[0],
      landmarks: pred.landmarks // eye/nose/ear positions for focus/drowsy
    }));
  };

  // Object (phone/books) detection
  const detectObjectsTF = async (videoElement) => {
    if (!cocoSsdModelRef.current || !videoElement) return [];
    const predictions = await cocoSsdModelRef.current.detect(videoElement);
    return predictions.filter(pred => ["cell phone","book"].includes(pred.class))
      .map(pred => ({
        class: pred.class === "cell phone" ? "phone" : "book",
        confidence: pred.score,
        bbox: pred.bbox
      }));
  };

  // Gaze/focus loss by looking away (using eye/nose landmarks or bounding box movement)
  const detectFocusLoss = (face) => {
    if (!face) {
      focusLostFrames.current++;
      return focusLostFrames.current >= FOCUS_LOST_FRAMES;
    }
    // If you want true gaze: require face landmarks detection (not just BlazeFace)
    // Simple: track bounding box center drifting away from center of the video
    const { centerX, centerY } = face;
    const videoW = videoRef.current?.videoWidth || 640;
    const videoH = videoRef.current?.videoHeight || 480;
    const dX = Math.abs(centerX - videoW / 2);
    const dY = Math.abs(centerY - videoH / 2);
    const drift = Math.sqrt(dX*dX + dY*dY);
    if (drift > Math.min(videoW, videoH) / 4) { focusLostFrames.current++; }
    else { focusLostFrames.current = Math.max(0, focusLostFrames.current-1); }
    return focusLostFrames.current >= FOCUS_LOST_FRAMES;
  };

  // Drowsiness: eyes closed (use landmarks for accurate, simulate EAR for now)
  const detectDrowsiness = (face) => {
    if (!face || !face.landmarks) {
      drowsyFrames.current = 0;
      return false;
    }
    // In production: calculate EAR (eye aspect ratio) from left/right eye landmarks
    // Here: we randomly simulate closed eyes 10% of the time for demo
    let isClosed = Math.random() < 0.08;
    drowsyFrames.current = isClosed ? drowsyFrames.current + 1 : 0;
    return drowsyFrames.current >= DROWSINESS_FRAMES;
  };

  // Main detection loop
  const runDetection = useCallback(async () => {
    if (!videoRef.current || !cameraActive || !modelsLoaded) return;
    const faces = await detectFacesTF(videoRef.current);
    const objects = await detectObjectsTF(videoRef.current);
    const detectionResults = [];

    const mainFace = faces[0];

    // Face Absent
    if (!mainFace) {
      if (!currentDetections.faceAbsent) triggerDetection('faceAbsent');
      detectionResults.push({ type: 'faceAbsent', active: true });
    } else {
      if (currentDetections.faceAbsent) clearDetection('faceAbsent');
      detectionResults.push({ type: 'faceAbsent', active: false });
    }

    // Multiple Faces
    if (faces.length > 1) {
      if (!currentDetections.multipleFaces) triggerDetection('multipleFaces');
      detectionResults.push({ type: 'multipleFaces', active: true });
    } else {
      if (currentDetections.multipleFaces) clearDetection('multipleFaces');
      detectionResults.push({ type: 'multipleFaces', active: false });
    }

    // Focus Lost
    const focusLost = detectFocusLoss(mainFace);
    if (focusLost) {
      if (!currentDetections.focusLost) triggerDetection('focusLost');
      detectionResults.push({ type: 'focusLost', active: true });
    } else {
      if (currentDetections.focusLost) clearDetection('focusLost');
      detectionResults.push({ type: 'focusLost', active: false });
    }

    // Drowsiness
    const isDrowsy = detectDrowsiness(mainFace);
    if (isDrowsy) {
      if (!currentDetections.drowsiness) triggerDetection('drowsiness');
      detectionResults.push({ type: 'drowsiness', active: true });
    } else {
      if (currentDetections.drowsiness) clearDetection('drowsiness');
      detectionResults.push({ type: 'drowsiness', active: false });
    }

    // Object detection (phone/books)
    let phoneDetected = objects.some(obj => obj.class === 'phone');
    let notesDetected = objects.some(obj => obj.class === 'book');

    if (phoneDetected) {
      if (!currentDetections.phoneDetected) triggerDetection('phoneDetected');
      detectionResults.push({ type: 'phoneDetected', active: true });
    } else {
      if (currentDetections.phoneDetected) clearDetection('phoneDetected');
      detectionResults.push({ type: 'phoneDetected', active: false });
    }
    if (notesDetected) {
      if (!currentDetections.notesDetected) triggerDetection('notesDetected');
      detectionResults.push({ type: 'notesDetected', active: true });
    } else {
      if (currentDetections.notesDetected) clearDetection('notesDetected');
      detectionResults.push({ type: 'notesDetected', active: false });
    }

    setDetectionResults(detectionResults);
  }, [cameraActive, modelsLoaded, currentDetections]);

  useEffect(() => {
    if (isMonitoring && cameraActive && modelsLoaded) {
      detectionIntervalRef.current = setInterval(runDetection, 300);
    } else if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current);
    }
    return () => {
      if (detectionIntervalRef.current) clearInterval(detectionIntervalRef.current);
    };
  }, [isMonitoring, cameraActive, modelsLoaded, runDetection]);

  const triggerDetection = (detectionType) => {
    setStats(prev => ({ ...prev, [detectionType]: prev[detectionType] + 1 }));
    setCurrentDetections(prev => ({ ...prev, [detectionType]: true }));
  };
  const clearDetection = detectionType => setCurrentDetections(prev => ({ ...prev, [detectionType]: false }));
  const resetStats = () => {
    setStats({ focusLost: 0, faceAbsent: 0, multipleFaces: 0, phoneDetected: 0, notesDetected: 0, drowsiness: 0 });
    setCurrentDetections({ focusLost: false, faceAbsent: false, multipleFaces: false, phoneDetected: false, notesDetected: false, drowsiness: false });
    focusLostFrames.current = drowsyFrames.current = 0;
  };

  const handleStartStop = async () => {
    if (!isMonitoring) {
      if (!cameraActive) await startCamera();
      setIsMonitoring(true);
    } else {
      setIsMonitoring(false); stopCamera();
    }
  };

  const icons = { focusLost: Eye, faceAbsent: EyeOff, multipleFaces: Users, phoneDetected: Smartphone, notesDetected: FileText, drowsiness: Moon };
  const labels = { focusLost: "Focus Lost", faceAbsent: "Face Absent", multipleFaces: "Multiple Faces", phoneDetected: "Phone Detected", notesDetected: "Notes Detected", drowsiness: "Drowsiness" };
  const getDetectionIcon = key => icons[key];
  const getDetectionLabel = key => labels[key];
  const getDetectionColor = (type, a) =>
    a ? 'text-red-600 bg-red-100'
      : {focusLost:'text-orange-600',faceAbsent:'text-red-600',multipleFaces:'text-yellow-600',phoneDetected:'text-purple-600',notesDetected:'text-blue-600',drowsiness:'text-indigo-600'}[type] || 'text-gray-600';
  const getTotalViolations = () => Object.values(stats).reduce((t,n)=>t+n,0);
  const getSeverityLevel = () => {
    const t=getTotalViolations(); if (t>=20) return {level:"Critical", color:"text-red-700 bg-red-200"};
    if (t>=15) return {level:"High Risk", color:"text-red-600 bg-red-100"};
    if (t>=8) return {level:"Medium Risk", color:"text-yellow-600 bg-yellow-100"};
    if (t>=3) return {level:"Low Risk", color:"text-orange-600 bg-orange-100"};
    return {level:"Normal",color:"text-green-600 bg-green-100"};
  };

  const severity = getSeverityLevel();

  return (
    <div className="max-w-7xl mx-auto p-6 bg-white">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Advanced Computer Vision Detection System
        </h1>
        <p className="text-gray-600">
          Real-time monitoring using TensorFlow.js for face/gaze/drowsiness/phone/notes.
        </p>
      </div>
      {loadingStatus && (
        <div className="mb-4 p-3 bg-blue-100 border border-blue-300 rounded-lg flex items-center">
          <Loader className="w-5 h-5 text-blue-600 animate-spin mr-2" />
          <p className="text-blue-800">{loadingStatus}</p>
        </div>
      )}
      <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
        <div className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
          <div className={`w-3 h-3 rounded-full ${modelsLoaded ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
          <span>Models: {modelsLoaded ? 'Ready' : 'Loading'}</span>
        </div>
        <div className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
          <div className={`w-3 h-3 rounded-full ${cameraActive ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span>Camera: {cameraActive ? 'Active' : 'Inactive'}</span>
        </div>
        <div className="flex items-center space-x-2 p-2 bg-gray-50 rounded">
          <div className={`w-3 h-3 rounded-full ${isMonitoring ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
          <span>Detection: {isMonitoring ? 'Running' : 'Stopped'}</span>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2 bg-gray-100 rounded-lg overflow-hidden">
          <div className="p-4 bg-gray-200 flex items-center justify-between">
            <h3 className="font-semibold text-gray-800">Live Camera Feed</h3>
          </div>
          <div className="relative">
            <video
              ref={videoRef}
              className="w-full h-80 object-cover bg-gray-900"
              autoPlay
              muted
              playsInline
            />
            {!cameraActive && (
              <div className="absolute inset-0 flex items-center justify-center bg-gray-800">
                <div className="text-center">
                  <CameraOff className="w-16 h-16 text-gray-400 mx-auto mb-2" />
                  <p className="text-gray-400">Camera Inactive</p>
                </div>
              </div>
            )}
            {cameraActive && (
              <div className="absolute top-2 right-2">
                <div className="flex items-center space-x-1 bg-red-600 text-white px-3 py-1 rounded-full text-sm">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                  <span>LIVE</span>
                </div>
              </div>
            )}
            {isMonitoring && (
              <div className="absolute bottom-2 left-2 right-2">
                <div className="bg-black bg-opacity-70 text-white p-2 rounded text-xs">
                  <div className="grid grid-cols-3 gap-2">
                    {detectionResults.slice(0, 6).map((result, index) => {
                      const Icon = getDetectionIcon(result.type);
                      return (
                        <div key={index} className={`flex items-center ${result.active ? 'text-red-400' : 'text-green-400'}`}>
                          <Icon className="w-4 h-4 mr-1"/>
                          <span>{getDetectionLabel(result.type).split(' ')[0]}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        <div className="space-y-6">
          <div className="p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold text-gray-800 mb-4">Detection Control</h3>
            <div className="space-y-4">
              <button
                onClick={handleStartStop}
                disabled={!modelsLoaded}
                className={`w-full px-4 py-3 rounded-lg font-medium transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed ${
                  isMonitoring
                    ? 'bg-red-600 text-white hover:bg-red-700'
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                <div className="flex items-center justify-center">
                  {isMonitoring ? (
                    <>
                      <AlertTriangle className="w-5 h-5 mr-2" />
                      Stop Monitoring
                    </>
                  ) : (
                    <>
                      <Camera className="w-5 h-5 mr-2" />
                      Start Monitoring
                    </>
                  )}
                </div>
              </button>
              <button
                onClick={resetStats}
                className="w-full px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
              >
                Reset All Statistics
              </button>
            </div>
          </div>
          <div className="space-y-3">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-blue-800 font-semibold mb-1">Total Violations</h3>
              <p className="text-3xl font-bold text-blue-900">{getTotalViolations()}</p>
              <p className="text-sm text-blue-700">Across all detection types</p>
            </div>
            <div className={`p-4 rounded-lg ${severity.color.replace('text-', 'bg-').replace('-600', '-50').replace('-700', '-50')}`}>
              <h3 className={`font-semibold mb-1 ${severity.color.split(' ')[0]}`}>Risk Assessment</h3>
              <p className={`text-xl font-bold ${severity.color.split(' ')[0].replace('-600', '-900').replace('-700', '-900')}`}>
                {severity.level}
              </p>
              <p className={`text-sm ${severity.color.split(' ')[0].replace('-600', '-700').replace('-700', '-700')}`}>
                {severity.level === 'Critical' && 'Immediate intervention required'}
                {severity.level === 'High Risk' && 'Close monitoring needed'}
                {severity.level === 'Medium Risk' && 'Attention recommended'}
                {severity.level === 'Low Risk' && 'Minor concerns detected'}
                {severity.level === 'Normal' && 'All systems normal'}
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h3 className="font-semibold text-gray-800 mb-2">Active Alerts</h3>
              <div className="space-y-2">
                {Object.entries(currentDetections).filter(([_, active]) => active).length === 0 ? (
                  <p className="text-green-600 text-sm">No active violations</p>
                ) : (
                  Object.entries(currentDetections)
                    .filter(([_, active]) => active)
                    .map(([key, _]) => (
                      <div key={key} className="flex items-center text-red-600 text-sm">
                        <AlertTriangle className="w-4 h-4 mr-2 animate-pulse" />
                        <span>{getDetectionLabel(key)} - Active</span>
                      </div>
                    ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
      {/* Detection Statistics */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 mb-4">Detection Statistics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(stats).map(([key, count]) => {
            const Icon = getDetectionIcon(key);
            const isActive = currentDetections[key];
            const colorClass = getDetectionColor(key, isActive);

            return (
              <div 
                key={key}
                className={`p-6 rounded-lg border-2 transition-all duration-300 ${
                  isActive 
                    ? 'border-red-300 bg-red-50 shadow-xl transform scale-105' 
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:shadow-md'
                }`}
              >
                <div className="flex items-center justify-between mb-4">
                  <div className={`p-3 rounded-lg ${isActive ? colorClass : 'bg-gray-100'}`}>
                    <Icon className={`w-7 h-7 ${isActive ? 'text-red-600' : colorClass}`} />
                  </div>
                  {isActive && (
                    <div className="flex items-center">
                      <AlertTriangle className="w-5 h-5 text-red-500 animate-pulse mr-1" />
                      <span className="text-xs text-red-600 font-medium animate-pulse">ACTIVE</span>
                    </div>
                  )}
                </div>
                <h3 className="font-semibold text-gray-800 mb-2 text-lg">
                  {getDetectionLabel(key)}
                </h3>
                <div className="flex items-end justify-between mb-2">
                  <span className="text-3xl font-bold text-gray-900">{count}</span>
                  <span className="text-sm text-gray-500">violations</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-300 ${
                      isActive ? 'bg-red-500' : 'bg-blue-500'
                    }`}
                    style={{ 
                      width: `${Math.min((count / Math.max(getTotalViolations(), 1)) * 100, 100)}%` 
                    }}
                  ></div>
                </div>
                <div className="text-xs text-gray-500">
                  {count > 0 && (
                    <span>
                      {((count / Math.max(getTotalViolations(), 1)) * 100).toFixed(1)}% of total violations
                    </span>
                  )}
                  {count === 0 && <span>No violations detected</span>}
                </div>
              </div>
            );
          })}
        </div>
      </div>
      {/* Info section */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg border border-blue-200">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">
          🔬 Advanced Computer Vision Features
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium text-green-800 mb-2">TensorFlow.js Integration:</h4>
            <ul className="space-y-1 text-gray-700">
              <li>• BlazeFace model for face/gaze</li>
              <li>• Drowsiness & FocusLost simulation</li>
              <li>• COCO-SSD for phone/book detection</li>
            </ul>
          </div>
        </div>
        <div className="mt-4 p-3 bg-yellow-100 border border-yellow-300 rounded">
          <p className="text-yellow-800 text-sm">
            <strong>Production Note:</strong> This demo uses simulated focus/eye detection for browser speed.
            Integrate face landmark models for robust gaze/drowsiness in real deployments.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DetectionStatistics;