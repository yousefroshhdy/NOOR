
import { supabase } from '@/integrations/supabase/client';
import { Database } from '@/integrations/supabase/types';

type Vehicle = Database['public']['Tables']['vehicles']['Row'];
type VehicleInsert = Database['public']['Tables']['vehicles']['Insert'];
type VehiclePosition = Database['public']['Tables']['vehicle_positions']['Row'];
type VehiclePositionInsert = Database['public']['Tables']['vehicle_positions']['Insert'];
type DrowsinessEvent = Database['public']['Tables']['drowsiness_events']['Row'];
type DrowsinessEventInsert = Database['public']['Tables']['drowsiness_events']['Insert'];
type DrivingSession = Database['public']['Tables']['driving_sessions']['Row'];

export const vehicleService = {
  // Vehicle CRUD operations
  async getVehicles(): Promise<Vehicle[]> {
    const { data, error } = await supabase
      .from('vehicles')
      .select('*')
      .eq('is_active', true)
      .order('created_at', { ascending: false });
    
    if (error) throw error;
    return data || [];
  },

  async getVehicleById(id: string): Promise<Vehicle | null> {
    const { data, error } = await supabase
      .from('vehicles')
      .select('*')
      .eq('id', id)
      .single();
    
    if (error) throw error;
    return data;
  },

  async createVehicle(vehicle: VehicleInsert): Promise<Vehicle> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) throw new Error('User not authenticated');

    const { data, error } = await supabase
      .from('vehicles')
      .insert({
        ...vehicle,
        user_id: user.id
      })
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  // GPS Position tracking
  async updateVehiclePosition(vehicleId: string, position: Omit<VehiclePositionInsert, 'vehicle_id'>): Promise<void> {
    const { error } = await supabase
      .from('vehicle_positions')
      .insert({
        vehicle_id: vehicleId,
        ...position
      });
    
    if (error) throw error;
  },

  async getVehiclePositions(vehicleId: string, limit = 100): Promise<VehiclePosition[]> {
    const { data, error } = await supabase
      .from('vehicle_positions')
      .select('*')
      .eq('vehicle_id', vehicleId)
      .order('timestamp', { ascending: false })
      .limit(limit);
    
    if (error) throw error;
    return data || [];
  },

  async getLatestVehiclePosition(vehicleId: string): Promise<VehiclePosition | null> {
    const { data, error } = await supabase
      .from('vehicle_positions')
      .select('*')
      .eq('vehicle_id', vehicleId)
      .order('timestamp', { ascending: false })
      .limit(1)
      .maybeSingle();
    
    if (error) throw error;
    return data;
  },

  // Drowsiness detection
  async recordDrowsinessEvent(event: Omit<DrowsinessEventInsert, 'user_id'>): Promise<void> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) throw new Error('User not authenticated');

    const { error } = await supabase
      .from('drowsiness_events')
      .insert({
        ...event,
        user_id: user.id
      });
    
    if (error) throw error;
  },

  async getDrowsinessEvents(vehicleId: string): Promise<DrowsinessEvent[]> {
    const { data, error } = await supabase
      .from('drowsiness_events')
      .select('*')
      .eq('vehicle_id', vehicleId)
      .order('timestamp', { ascending: false });
    
    if (error) throw error;
    return data || [];
  },

  // Driving sessions
  async startDrivingSession(vehicleId: string): Promise<DrivingSession> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) throw new Error('User not authenticated');

    const { data, error } = await supabase
      .from('driving_sessions')
      .insert({
        vehicle_id: vehicleId,
        user_id: user.id,
        is_active: true
      })
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  async endDrivingSession(sessionId: string, stats: {
    total_distance?: number;
    max_speed?: number;
    avg_speed?: number;
    drowsiness_alerts?: number;
  }): Promise<void> {
    const { error } = await supabase
      .from('driving_sessions')
      .update({
        end_time: new Date().toISOString(),
        is_active: false,
        ...stats
      })
      .eq('id', sessionId);
    
    if (error) throw error;
  },

  // Real-time subscriptions
  subscribeToVehiclePositions(vehicleId: string, callback: (position: VehiclePosition) => void) {
    return supabase
      .channel(`vehicle-positions-${vehicleId}`)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'vehicle_positions',
          filter: `vehicle_id=eq.${vehicleId}`
        },
        (payload) => callback(payload.new as VehiclePosition)
      )
      .subscribe();
  },

  subscribeToAllVehiclePositions(callback: (position: VehiclePosition) => void) {
    return supabase
      .channel('all-vehicle-positions')
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'vehicle_positions'
        },
        (payload) => callback(payload.new as VehiclePosition)
      )
      .subscribe();
  },

  subscribeToDrowsinessEvents(vehicleId: string, callback: (event: DrowsinessEvent) => void) {
    return supabase
      .channel(`drowsiness-events-${vehicleId}`)
      .on(
        'postgres_changes',
        {
          event: 'INSERT',
          schema: 'public',
          table: 'drowsiness_events',
          filter: `vehicle_id=eq.${vehicleId}`
        },
        (payload) => callback(payload.new as DrowsinessEvent)
      )
      .subscribe();
  }
};
