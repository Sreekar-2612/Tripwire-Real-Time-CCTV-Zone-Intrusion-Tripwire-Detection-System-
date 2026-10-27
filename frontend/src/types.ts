export type Point2D = [number, number]; // [x, y] normalized 0.0 - 1.0

export interface Zone {
  id?: string;
  stream_id?: string;
  name: string;
  zone_type: 'polygon' | 'tripwire';
  coordinates: Point2D[];
  armed: boolean;
  color: string;
  schedule_start?: string | null;
  schedule_end?: string | null;
  created_at?: string;
}

export interface Stream {
  id: string;
  name: string;
  source_type: 'webcam' | 'rtsp' | 'file';
  source_url: string;
  is_active: boolean;
  loop_file: boolean;
  fps: number;
  width: number;
  height: number;
  created_at: string;
  zones?: Zone[];
}

export interface EventItem {
  id: string;
  stream_id: string;
  zone_id?: string | null;
  zone_name?: string | null;
  track_id: number;
  event_type: 'ENTER' | 'EXIT' | 'LOITER' | 'CROSS_IN' | 'CROSS_OUT' | 'TEST_ALERT';
  timestamp: string;
  confidence: number;
  snapshot_url?: string | null;
  clip_url?: string | null;
  details?: string | null;
}

export interface Settings {
  confidence_threshold: number;
  cooldown_seconds: number;
  loitering_seconds: number;
  debounce_frames: number;
  frame_skip: number;
  telegram_bot_token?: string | null;
  telegram_chat_id?: string | null;
  webhook_url?: string | null;
  smtp_host?: string | null;
  smtp_port: number;
  smtp_user?: string | null;
  smtp_password?: string | null;
  alert_email_to?: string | null;
  alert_email_from?: string | null;
  enable_audio_alert: boolean;
}

export interface WSAlertMessage {
  type: string;
  event_id: string;
  stream_id: string;
  stream_name?: string;
  zone_id?: string | null;
  zone_name?: string | null;
  track_id: number;
  event_type: string;
  timestamp: string;
  confidence: number;
  snapshot_url?: string | null;
  clip_url?: string | null;
  message: string;
}
