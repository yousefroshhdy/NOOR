
import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated, logout, getCurrentUser } from '@/services/authService';
import { useTheme } from '@/contexts/ThemeContext';
import AppLayout from '@/components/layout/AppLayout';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import { Bell, Car, MapPin, AlertTriangle, LogOut, Moon, Sun } from 'lucide-react';

const Settings = () => {
  const navigate = useNavigate();
  const user = getCurrentUser();
  const { theme, toggleTheme } = useTheme();

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }
  }, [navigate]);

  const handleLogout = () => {
    logout();
    navigate('/auth');
  };

  return (
    <AppLayout title="Settings">
      <div className="max-w-4xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">Settings</h1>
        
        {/* Account Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Account</CardTitle>
            <CardDescription>Manage your account settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 rounded-full bg-gray-200 overflow-hidden">
                {user?.avatar ? (
                  <img src={user.avatar} alt={user.name} className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-drivepulse-teal text-white font-medium text-xl">
                    {user?.name?.charAt(0)}
                  </div>
                )}
              </div>
              <div>
                <h3 className="font-medium">{user?.name}</h3>
                <p className="text-sm text-gray-500">{user?.email}</p>
              </div>
            </div>
            
            <Separator />
            
            <div className="grid gap-4">
              <Button variant="outline" className="justify-start">
                Edit Profile
              </Button>
              <Button variant="outline" className="justify-start">
                Change Password
              </Button>
              <Button 
                variant="destructive" 
                className="justify-start" 
                onClick={handleLogout}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Sign Out
              </Button>
            </div>
          </CardContent>
        </Card>
        
        {/* Appearance Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Appearance</CardTitle>
            <CardDescription>Customize the app's look and feel</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {theme === 'dark' ? (
                  <Moon className="h-4 w-4 text-gray-500" />
                ) : (
                  <Sun className="h-4 w-4 text-gray-500" />
                )}
                <span>Dark Mode</span>
              </div>
              <Switch checked={theme === 'dark'} onCheckedChange={toggleTheme} />
            </div>
          </CardContent>
        </Card>
        
        {/* Notification Settings */}
        <Card>
          <CardHeader>
            <CardTitle>Notifications</CardTitle>
            <CardDescription>Configure how you receive alerts</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Bell className="h-4 w-4 text-gray-500" />
                  <span>Push Notifications</span>
                </div>
                <Switch defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MapPin className="h-4 w-4 text-gray-500" />
                  <span>Location Updates</span>
                </div>
                <Switch defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Car className="h-4 w-4 text-gray-500" />
                  <span>Vehicle Status Changes</span>
                </div>
                <Switch defaultChecked />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-gray-500" />
                  <span>Critical Alerts</span>
                </div>
                <Switch defaultChecked />
              </div>
            </div>
            
            <Separator />
            
            <div>
              <p className="text-sm font-medium mb-2">Alert Sensitivity</p>
              <ToggleGroup type="single" defaultValue="medium">
                <ToggleGroupItem value="low">Low</ToggleGroupItem>
                <ToggleGroupItem value="medium">Medium</ToggleGroupItem>
                <ToggleGroupItem value="high">High</ToggleGroupItem>
              </ToggleGroup>
            </div>
          </CardContent>
        </Card>
        
        {/* App Settings */}
        <Card>
          <CardHeader>
            <CardTitle>App Settings</CardTitle>
            <CardDescription>Customize your experience</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span>Map View</span>
              <ToggleGroup type="single" defaultValue="streets">
                <ToggleGroupItem value="streets">Streets</ToggleGroupItem>
                <ToggleGroupItem value="satellite">Satellite</ToggleGroupItem>
              </ToggleGroup>
            </div>
            
            <div className="flex items-center justify-between">
              <span>Distance Units</span>
              <ToggleGroup type="single" defaultValue="miles">
                <ToggleGroupItem value="miles">Miles</ToggleGroupItem>
                <ToggleGroupItem value="kilometers">Kilometers</ToggleGroupItem>
              </ToggleGroup>
            </div>
            
            <div className="flex items-center justify-between">
              <span>Real-time Updates</span>
              <Switch defaultChecked />
            </div>
          </CardContent>
        </Card>
        
        <div className="text-center text-sm text-gray-500 pt-4 pb-16">
          DrivePulse Tracker v1.0.0
        </div>
      </div>
    </AppLayout>
  );
};

export default Settings;
