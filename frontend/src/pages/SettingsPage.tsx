import React, { useState, useEffect } from 'react';
import type { Settings } from '../types';
import { fetchSettings, updateSettings, testAlertChannel } from '../api/client';
import { Sliders, Send, CheckCircle, AlertCircle, Cpu, Mail, MessageSquare, Globe, Save } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; message: string } | null>(null);
  const [testingChannel, setTestingChannel] = useState<string | null>(null);

  useEffect(() => {
    fetchSettings()
      .then(setSettings)
      .catch(() => {
        setFeedback({ type: 'error', message: 'Failed to load system settings' });
      });
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    setIsSaving(true);
    setFeedback(null);

    try {
      const updated = await updateSettings(settings);
      setSettings(updated);
      setFeedback({ type: 'success', message: 'System settings successfully updated & applied!' });
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.message || 'Failed to save settings' });
    } finally {
      setIsSaving(false);
    }
  };

  const handleTestAlert = async (channel: 'telegram' | 'webhook' | 'email') => {
    setTestingChannel(channel);
    setFeedback(null);
    try {
      const res = await testAlertChannel(channel);
      setFeedback({ type: 'success', message: res.message });
    } catch (err: any) {
      setFeedback({ type: 'error', message: err.message || `Failed to send ${channel} test alert.` });
    } finally {
      setTestingChannel(null);
    }
  };

  if (!settings) {
    return <div className="p-8 text-center text-slate-500 font-mono">Loading settings...</div>;
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
        <div className="flex items-center space-x-3">
          <Sliders className="w-6 h-6 text-red-500" />
          <div>
            <h2 className="text-xl font-bold text-white tracking-wide">System & Alert Settings</h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Tune AI detection parameters, event debounce, and configure external alert dispatch channels.
            </p>
          </div>
        </div>
      </div>

      {feedback && (
        <div
          className={`p-4 rounded-xl border flex items-center space-x-3 text-xs font-mono animate-in fade-in duration-200 ${
            feedback.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400'
              : 'bg-red-500/10 border-red-500/40 text-red-400'
          }`}
        >
          {feedback.type === 'success' ? <CheckCircle className="w-4 h-4 flex-shrink-0" /> : <AlertCircle className="w-4 h-4 flex-shrink-0" />}
          <span>{feedback.message}</span>
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Computer Vision & State Machine Section */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-5">
          <div className="flex items-center space-x-2 text-white border-b border-slate-800 pb-3">
            <Cpu className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold uppercase tracking-wider">Detection & State Machine Tuning</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Confidence Slider */}
            <div>
              <div className="flex items-center justify-between text-xs font-mono mb-2">
                <span className="text-slate-300">YOLO Confidence Threshold:</span>
                <span className="font-bold text-red-400">{(settings.confidence_threshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.10"
                max="0.95"
                step="0.05"
                value={settings.confidence_threshold}
                onChange={(e) => setSettings({ ...settings, confidence_threshold: parseFloat(e.target.value) })}
                className="w-full accent-red-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Lower detects people from further away; higher eliminates false positives.
              </p>
            </div>

            {/* Debounce frames */}
            <div>
              <div className="flex items-center justify-between text-xs font-mono mb-2">
                <span className="text-slate-300">Debounce Inside Frames:</span>
                <span className="font-bold text-red-400">{settings.debounce_frames} frames</span>
              </div>
              <input
                type="range"
                min="1"
                max="8"
                step="1"
                value={settings.debounce_frames}
                onChange={(e) => setSettings({ ...settings, debounce_frames: parseInt(e.target.value) })}
                className="w-full accent-red-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Consecutive frames inside zone before triggering an ENTER alarm (prevents edge jitter).
              </p>
            </div>

            {/* Cooldown Seconds */}
            <div>
              <div className="flex items-center justify-between text-xs font-mono mb-2">
                <span className="text-slate-300">Per-Track Cooldown:</span>
                <span className="font-bold text-red-400">{settings.cooldown_seconds}s</span>
              </div>
              <input
                type="range"
                min="1.0"
                max="60.0"
                step="1.0"
                value={settings.cooldown_seconds}
                onChange={(e) => setSettings({ ...settings, cooldown_seconds: parseFloat(e.target.value) })}
                className="w-full accent-red-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Minimum seconds before the same person can trigger a repeat alarm.
              </p>
            </div>

            {/* Loitering Alert Seconds */}
            <div>
              <div className="flex items-center justify-between text-xs font-mono mb-2">
                <span className="text-slate-300">Loitering Threshold:</span>
                <span className="font-bold text-red-400">{settings.loitering_seconds}s</span>
              </div>
              <input
                type="range"
                min="2.0"
                max="120.0"
                step="1.0"
                value={settings.loitering_seconds}
                onChange={(e) => setSettings({ ...settings, loitering_seconds: parseFloat(e.target.value) })}
                className="w-full accent-red-500 cursor-pointer"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Fires a LOITER alarm if a person remains inside a protected region beyond this duration.
              </p>
            </div>
          </div>
        </div>

        {/* Telegram Bot Setup */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-white">
              <MessageSquare className="w-4 h-4 text-blue-400" />
              <h3 className="text-sm font-bold uppercase tracking-wider">Telegram Bot Alerts</h3>
            </div>
            <button
              type="button"
              disabled={testingChannel === 'telegram'}
              onClick={() => handleTestAlert('telegram')}
              className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 border border-blue-500/30 text-xs font-mono cursor-pointer disabled:opacity-50"
            >
              <Send className="w-3 h-3" />
              <span>{testingChannel === 'telegram' ? 'Sending...' : 'Send Test Alert'}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
            <div>
              <label className="text-slate-400 block mb-1">Bot API Token</label>
              <input
                type="text"
                placeholder="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
                value={settings.telegram_bot_token || ''}
                onChange={(e) => setSettings({ ...settings, telegram_bot_token: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
              />
            </div>
            <div>
              <label className="text-slate-400 block mb-1">Target Chat ID</label>
              <input
                type="text"
                placeholder="-100123456789 or @channel"
                value={settings.telegram_chat_id || ''}
                onChange={(e) => setSettings({ ...settings, telegram_chat_id: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
              />
            </div>
          </div>
        </div>

        {/* Generic Webhook Setup */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-white">
              <Globe className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold uppercase tracking-wider">Generic Webhook (HTTP POST)</h3>
            </div>
            <button
              type="button"
              disabled={testingChannel === 'webhook'}
              onClick={() => handleTestAlert('webhook')}
              className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-emerald-600/20 hover:bg-emerald-600/30 text-emerald-400 border border-emerald-500/30 text-xs font-mono cursor-pointer disabled:opacity-50"
            >
              <Send className="w-3 h-3" />
              <span>{testingChannel === 'webhook' ? 'Sending...' : 'Send Test Ping'}</span>
            </button>
          </div>

          <div className="text-xs font-mono">
            <label className="text-slate-400 block mb-1">Webhook Endpoint URL</label>
            <input
              type="url"
              placeholder="https://your-server.com/api/cctv-webhooks"
              value={settings.webhook_url || ''}
              onChange={(e) => setSettings({ ...settings, webhook_url: e.target.value })}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
            />
          </div>
        </div>

        {/* Email Setup */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center space-x-2 text-white">
              <Mail className="w-4 h-4 text-purple-400" />
              <h3 className="text-sm font-bold uppercase tracking-wider">SMTP Email Alerts</h3>
            </div>
            <button
              type="button"
              disabled={testingChannel === 'email'}
              onClick={() => handleTestAlert('email')}
              className="flex items-center space-x-1.5 px-3 py-1 rounded-lg bg-purple-600/20 hover:bg-purple-600/30 text-purple-400 border border-purple-500/30 text-xs font-mono cursor-pointer disabled:opacity-50"
            >
              <Send className="w-3 h-3" />
              <span>{testingChannel === 'email' ? 'Sending...' : 'Send Test Email'}</span>
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs font-mono">
            <div>
              <label className="text-slate-400 block mb-1">SMTP Host</label>
              <input
                type="text"
                placeholder="smtp.gmail.com"
                value={settings.smtp_host || ''}
                onChange={(e) => setSettings({ ...settings, smtp_host: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
              />
            </div>
            <div>
              <label className="text-slate-400 block mb-1">SMTP Port</label>
              <input
                type="number"
                value={settings.smtp_port}
                onChange={(e) => setSettings({ ...settings, smtp_port: parseInt(e.target.value) || 587 })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
              />
            </div>
            <div>
              <label className="text-slate-400 block mb-1">Recipient Email</label>
              <input
                type="email"
                placeholder="security-desk@company.com"
                value={settings.alert_email_to || ''}
                onChange={(e) => setSettings({ ...settings, alert_email_to: e.target.value })}
                className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3.5 py-2 text-white focus:outline-none focus:border-red-500"
              />
            </div>
          </div>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end pt-2">
          <button
            type="submit"
            disabled={isSaving}
            className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-xs uppercase tracking-wider shadow-xl shadow-red-950/80 transition-all cursor-pointer disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            <span>{isSaving ? 'Applying Settings...' : 'Save & Apply All Settings'}</span>
          </button>
        </div>
      </form>
    </div>
  );
};
