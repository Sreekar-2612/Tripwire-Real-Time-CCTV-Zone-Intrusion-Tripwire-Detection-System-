import React, { useEffect, useState } from 'react';
import { AlertTriangle, X, Eye } from 'lucide-react';
import type { WSAlertMessage } from '../types';

interface AlarmToastProps {
  alert: WSAlertMessage | null;
  onDismiss: () => void;
  onViewSnapshot?: (url: string) => void;
}

export const AlarmToast: React.FC<AlarmToastProps> = ({ alert, onDismiss, onViewSnapshot }) => {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    if (alert && alert.event_type !== 'EXIT') {
      setVisible(true);
      const timer = setTimeout(() => {
        setVisible(false);
      }, 7000);
      return () => clearTimeout(timer);
    }
  }, [alert]);

  if (!visible || !alert) return null;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-md w-full animate-in fade-in slide-in-from-bottom-5 duration-300">
      <div className="p-4 rounded-xl bg-slate-900/95 border-2 border-red-500 shadow-2xl shadow-red-950/80 text-white backdrop-blur-xl animate-alarm-pulse">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-lg bg-red-600/20 border border-red-500/50 text-red-400">
              <AlertTriangle className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="text-xs font-bold uppercase tracking-widest px-2 py-0.5 rounded bg-red-600 text-white">
                  {alert.event_type} ALARM
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Track #{alert.track_id}
                </span>
              </div>
              <h4 className="text-base font-bold text-white mt-1">
                Zone Intrusion: {alert.zone_name || 'Protected Zone'}
              </h4>
              <p className="text-xs text-slate-300 mt-0.5 font-mono">
                {alert.message}
              </p>
            </div>
          </div>

          <button
            onClick={() => {
              setVisible(false);
              onDismiss();
            }}
            className="text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {alert.snapshot_url && (
          <div className="mt-3 relative rounded-lg overflow-hidden border border-slate-700 bg-black max-h-36 flex items-center justify-center group cursor-pointer"
               onClick={() => onViewSnapshot && onViewSnapshot(alert.snapshot_url!)}>
            <img
              src={alert.snapshot_url}
              alt="Intrusion Snapshot"
              className="w-full h-36 object-cover opacity-90 group-hover:opacity-100 transition-opacity"
            />
            <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
              <span className="flex items-center space-x-1.5 text-xs font-medium text-white px-3 py-1.5 rounded-md bg-red-600/90">
                <Eye className="w-3.5 h-3.5" />
                <span>View Full Snapshot</span>
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
