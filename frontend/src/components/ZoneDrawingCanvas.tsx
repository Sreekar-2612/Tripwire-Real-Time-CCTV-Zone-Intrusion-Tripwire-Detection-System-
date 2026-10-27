import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Pentagon, TrendingUp, Trash2, Check, RotateCcw, Shield, ShieldOff } from 'lucide-react';
import type { Zone, Point2D } from '../types';

interface ZoneDrawingCanvasProps {
  videoUrl: string;
  zones: Zone[];
  onSaveZones: (zones: Zone[]) => void;
  isSaving?: boolean;
  isAlarmActive?: boolean;
}

export const ZoneDrawingCanvas: React.FC<ZoneDrawingCanvasProps> = ({
  videoUrl,
  zones: initialZones,
  onSaveZones,
  isSaving = false,
  isAlarmActive = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [zones, setZones] = useState<Zone[]>(initialZones);
  const [selectedZoneIndex, setSelectedZoneIndex] = useState<number | null>(null);
  const [mode, setMode] = useState<'idle' | 'polygon' | 'tripwire' | 'dragging'>('idle');
  const [currentPoints, setCurrentPoints] = useState<Point2D[]>([]);
  const [draggedPoint, setDraggedPoint] = useState<{ zoneIdx: number; pointIdx: number } | null>(null);
  const [isDirty, setIsDirty] = useState(false);
  const [videoKey, setVideoKey] = useState(Date.now());

  useEffect(() => {
    setZones(initialZones);
    setIsDirty(false);
  }, [initialZones]);

  // Convert mouse event coordinates to normalized [0.0, 1.0]
  const getNormalizedPoint = useCallback((e: React.MouseEvent<HTMLCanvasElement>): Point2D => {
    const canvas = canvasRef.current;
    if (!canvas) return [0, 0];
    const rect = canvas.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    return [Math.round(x * 1000) / 1000, Math.round(y * 1000) / 1000];
  }, []);

  // Redraw canvas overlay (transparent background, only drawing zone strokes/fills)
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Render existing zones
    zones.forEach((z, zIdx) => {
      if (!z.coordinates || z.coordinates.length < 2) return;
      const isSelected = selectedZoneIndex === zIdx;
      const strokeColor = z.armed ? (z.color || '#EF4444') : '#94A3B8';

      ctx.beginPath();
      const [startX, startY] = z.coordinates[0];
      ctx.moveTo(startX * w, startY * h);

      for (let i = 1; i < z.coordinates.length; i++) {
        const [px, py] = z.coordinates[i];
        ctx.lineTo(px * w, py * h);
      }

      if (z.zone_type === 'polygon' && z.coordinates.length >= 3) {
        ctx.closePath();
        // Fill semi-transparent
        ctx.fillStyle = z.armed ? `${strokeColor}33` : 'rgba(148, 163, 184, 0.15)';
        ctx.fill();
      }

      ctx.lineWidth = isSelected ? 3 : 2;
      ctx.strokeStyle = strokeColor;
      ctx.setLineDash(z.armed ? [] : [6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw point handles
      z.coordinates.forEach(([px, py]) => {
        ctx.beginPath();
        ctx.arc(px * w, py * h, isSelected ? 6 : 4, 0, Math.PI * 2);
        ctx.fillStyle = isSelected ? '#FFFFFF' : strokeColor;
        ctx.fill();
        ctx.strokeStyle = '#0F172A';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });

      // Label
      const [lx, ly] = z.coordinates[0];
      ctx.font = 'bold 12px monospace';
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(`${z.name} (${z.armed ? 'ARMED' : 'DISARMED'})`, lx * w + 8, Math.max(16, ly * h - 8));
    });

    // Render ongoing in-progress drawing points
    if (currentPoints.length > 0) {
      ctx.beginPath();
      const [sx, sy] = currentPoints[0];
      ctx.moveTo(sx * w, sy * h);

      for (let i = 1; i < currentPoints.length; i++) {
        const [cx, cy] = currentPoints[i];
        ctx.lineTo(cx * w, cy * h);
      }

      ctx.strokeStyle = '#38BDF8';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
      ctx.setLineDash([]);

      currentPoints.forEach(([cx, cy]) => {
        ctx.beginPath();
        ctx.arc(cx * w, cy * h, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#38BDF8';
        ctx.fill();
      });
    }
  }, [zones, selectedZoneIndex, currentPoints]);

  // Adjust canvas pixel resolution to container
  useEffect(() => {
    const updateSize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (canvas && container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        renderCanvas();
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [renderCanvas]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Handle canvas mouse clicks for point creation
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = getNormalizedPoint(e);

    if (mode === 'polygon') {
      const nextPts = [...currentPoints, pt];
      // Check if closing polygon (clicked near starting point)
      if (nextPts.length >= 3) {
        const [sx, sy] = nextPts[0];
        const dist = Math.hypot(pt[0] - sx, pt[1] - sy);
        if (dist < 0.04) {
          // Finish polygon
          finishNewZone(nextPts.slice(0, -1), 'polygon');
          return;
        }
      }
      setCurrentPoints(nextPts);
    } else if (mode === 'tripwire') {
      const nextPts = [...currentPoints, pt];
      if (nextPts.length >= 2) {
        finishNewZone(nextPts, 'tripwire');
      } else {
        setCurrentPoints(nextPts);
      }
    } else if (mode === 'idle') {
      // Check if clicked near an existing zone handle to select it
      const canvas = canvasRef.current;
      if (!canvas) return;
      const w = canvas.width;
      const h = canvas.height;

      let hitZone: number | null = null;
      for (let zi = 0; zi < zones.length; zi++) {
        for (const [px, py] of zones[zi].coordinates) {
          const dist = Math.hypot(pt[0] * w - px * w, pt[1] * h - py * h);
          if (dist < 12) {
            hitZone = zi;
            break;
          }
        }
        if (hitZone !== null) break;
      }
      setSelectedZoneIndex(hitZone);
    }
  };

  const finishNewZone = (coords: Point2D[], type: 'polygon' | 'tripwire') => {
    const newZ: Zone = {
      name: `Zone ${zones.length + 1}`,
      zone_type: type,
      coordinates: coords,
      armed: true,
      color: type === 'tripwire' ? '#3B82F6' : '#EF4444',
    };
    const updated = [...zones, newZ];
    setZones(updated);
    setSelectedZoneIndex(updated.length - 1);
    setCurrentPoints([]);
    setMode('idle');
    setIsDirty(true);
  };

  // Dragging vertex handles
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'idle') return;
    const pt = getNormalizedPoint(e);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const w = canvas.width;
    const h = canvas.height;

    for (let zi = 0; zi < zones.length; zi++) {
      for (let pi = 0; pi < zones[zi].coordinates.length; pi++) {
        const [px, py] = zones[zi].coordinates[pi];
        const dist = Math.hypot(pt[0] * w - px * w, pt[1] * h - py * h);
        if (dist < 10) {
          setDraggedPoint({ zoneIdx: zi, pointIdx: pi });
          setSelectedZoneIndex(zi);
          return;
        }
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!draggedPoint) return;
    const pt = getNormalizedPoint(e);
    const updated = [...zones];
    const targetZone = { ...updated[draggedPoint.zoneIdx] };
    const nextCoords = [...targetZone.coordinates];
    nextCoords[draggedPoint.pointIdx] = pt;
    targetZone.coordinates = nextCoords;
    updated[draggedPoint.zoneIdx] = targetZone;
    setZones(updated);
    setIsDirty(true);
  };

  const handleMouseUp = () => {
    if (draggedPoint) {
      setDraggedPoint(null);
    }
  };

  const handleDeleteZone = (idx: number) => {
    const updated = zones.filter((_, i) => i !== idx);
    setZones(updated);
    if (selectedZoneIndex === idx) setSelectedZoneIndex(null);
    setIsDirty(true);
  };

  const handleToggleArmed = (idx: number) => {
    const updated = [...zones];
    updated[idx] = { ...updated[idx], armed: !updated[idx].armed };
    setZones(updated);
    setIsDirty(true);
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 w-full">
      {/* Left Column: Live Video + Transparent Canvas Overlay + Bottom Toolbar */}
      <div className="flex-1 flex flex-col space-y-3">
        <div
          ref={containerRef}
          className={`relative w-full aspect-video rounded-2xl overflow-hidden bg-black flex items-center justify-center transition-all ${
            isAlarmActive
              ? 'border-4 border-red-500 shadow-2xl shadow-red-950/90 animate-alarm-pulse'
              : 'border border-slate-800 shadow-2xl'
          }`}
        >
          {/* Live MJPEG Video Stream */}
          <img
            src={`${videoUrl}&k=${videoKey}`}
            alt="Live CCTV Camera Feed"
            className="w-full h-full object-contain pointer-events-none select-none"
            onError={() => {
              setTimeout(() => setVideoKey(Date.now()), 2500);
            }}
          />

          {/* Interactive Drawing Canvas (Transparent Overlay) */}
          <canvas
            ref={canvasRef}
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            className={`absolute inset-0 z-20 w-full h-full ${
              mode === 'idle' ? 'cursor-default' : 'cursor-crosshair'
            }`}
          />

          {/* Mode Guidance Tooltip Overlay */}
          {mode !== 'idle' && (
            <div className="absolute top-4 left-4 z-30 px-3.5 py-2 rounded-xl bg-slate-900/90 border border-cyan-500/50 text-xs font-mono text-cyan-400 flex items-center space-x-2.5 shadow-xl backdrop-blur-md">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-ping" />
              <span>
                {mode === 'polygon'
                  ? `Click to add vertices (${currentPoints.length} points). Double click or click start to close.`
                  : `Click 2 points across screen to define tripwire line (${currentPoints.length}/2)`}
              </span>
              <button
                onClick={() => {
                  setMode('idle');
                  setCurrentPoints([]);
                }}
                className="ml-3 text-slate-400 hover:text-white underline font-semibold cursor-pointer"
              >
                Cancel
              </button>
            </div>
          )}
        </div>

        {/* Toolbar Directly Below Video */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3.5 rounded-2xl glass-panel">
          <div className="flex items-center space-x-2.5">
            <button
              onClick={() => {
                setMode('polygon');
                setCurrentPoints([]);
              }}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                mode === 'polygon'
                  ? 'bg-red-600 text-white shadow-lg shadow-red-900/60 ring-2 ring-red-400'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <Pentagon className="w-4 h-4" />
              <span>+ Draw Polygon Zone</span>
            </button>

            <button
              onClick={() => {
                setMode('tripwire');
                setCurrentPoints([]);
              }}
              className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                mode === 'tripwire'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/60 ring-2 ring-blue-400'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span>+ Draw Directional Tripwire</span>
            </button>
          </div>

          <div className="flex items-center space-x-2.5">
            {isDirty && (
              <button
                onClick={() => {
                  setZones(initialZones);
                  setIsDirty(false);
                }}
                className="flex items-center space-x-1.5 px-3 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white bg-slate-800/80 hover:bg-slate-700 transition-colors cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Revert Changes</span>
              </button>
            )}

            <button
              disabled={isSaving || !isDirty}
              onClick={() => {
                onSaveZones(zones);
                setIsDirty(false);
              }}
              className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-bold transition-all cursor-pointer ${
                isDirty
                  ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-950/60'
                  : 'bg-slate-800/60 text-slate-500 cursor-not-allowed'
              }`}
            >
              <Check className="w-4 h-4" />
              <span>{isSaving ? 'Saving...' : 'Apply & Arm Zones'}</span>
            </button>
          </div>
        </div>
      </div>

      {/* Right Column: Configured Zones List Side Panel */}
      <div className="w-full lg:w-80 flex flex-col glass-panel rounded-2xl p-4 border border-slate-800 flex-shrink-0">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center space-x-2">
            <Shield className="w-4 h-4 text-red-500" />
            <span>Configured Zones ({zones.length})</span>
          </h4>
        </div>

        <div className="mt-3 space-y-2.5 max-h-[460px] overflow-y-auto pr-1">
          {zones.length === 0 ? (
            <div className="py-12 text-center text-xs text-slate-500 font-mono">
              No zones configured yet.<br />Click "+ Draw Polygon Zone" to define an intrusion boundary.
            </div>
          ) : (
            zones.map((z, idx) => (
              <div
                key={idx}
                onClick={() => setSelectedZoneIndex(idx)}
                className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                  selectedZoneIndex === idx
                    ? 'bg-slate-800/90 border-red-500/70 shadow-md shadow-red-950/30'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <span
                      className="w-3.5 h-3.5 rounded-full flex-shrink-0 shadow"
                      style={{ backgroundColor: z.color || '#EF4444' }}
                    />
                    <input
                      type="text"
                      value={z.name}
                      onChange={(e) => {
                        const updated = [...zones];
                        updated[idx].name = e.target.value;
                        setZones(updated);
                        setIsDirty(true);
                      }}
                      className="text-xs font-bold text-white bg-transparent border-b border-transparent focus:border-slate-500 focus:outline-none px-1"
                    />
                  </div>

                  <div className="flex items-center space-x-1.5">
                    {/* Armed toggle button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleToggleArmed(idx);
                      }}
                      title={z.armed ? 'Zone Armed' : 'Zone Disarmed'}
                      className={`p-1.5 rounded-lg text-xs transition-colors cursor-pointer ${
                        z.armed
                          ? 'bg-red-500/10 text-red-400 hover:bg-red-500/20'
                          : 'bg-slate-800 text-slate-500 hover:text-slate-300'
                      }`}
                    >
                      {z.armed ? <Shield className="w-3.5 h-3.5" /> : <ShieldOff className="w-3.5 h-3.5" />}
                    </button>

                    {/* Delete button */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteZone(idx);
                      }}
                      title="Delete zone"
                      className="p-1.5 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-slate-800 transition-colors cursor-pointer"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                <div className="mt-2.5 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span className="capitalize">{z.zone_type}</span>
                  <span>{z.coordinates.length} vertices</span>
                  <span className={z.armed ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
                    {z.armed ? 'ARMED' : 'DISARMED'}
                  </span>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
