
import { ReactNode, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import MobileNavbar from './MobileNavbar';
import { Menu, X } from 'lucide-react';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import Sidebar from './Sidebar';

interface AppLayoutProps {
  children: ReactNode;
  title?: string;
}

const AppLayout = ({ children, title = 'DrivePulse' }: AppLayoutProps) => {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  // Check if user is authenticated
  useEffect(() => {
    if (!isAuthenticated()) {
      navigate('/auth');
    }
  }, [navigate]);

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Mobile Header */}
      <header className="fixed top-0 left-0 right-0 h-14 bg-white shadow-sm z-40 md:hidden flex items-center px-4">
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center">
            <Sheet open={open} onOpenChange={setOpen}>
              <SheetTrigger asChild>
                <button className="p-2 rounded-md text-gray-700 hover:bg-gray-100">
                  {open ? <X size={20} /> : <Menu size={20} />}
                </button>
              </SheetTrigger>
              <SheetContent side="left" className="p-0 w-[240px]">
                <ScrollArea className="h-full">
                  <Sidebar />
                </ScrollArea>
              </SheetContent>
            </Sheet>
            <h1 className="ml-4 text-lg font-semibold text-gray-900">{title}</h1>
          </div>
        </div>
      </header>

      {/* Desktop Sidebar - Hidden on Mobile */}
      <div className="hidden md:block w-64 fixed inset-y-0">
        <div className="h-full bg-white border-r">
          <Sidebar />
        </div>
      </div>

      {/* Main Content Area */}
      <main className="flex-1 md:ml-64 pt-14 md:pt-0">
        <div className="h-full px-4 py-6 overflow-y-auto">
          {children}
        </div>
      </main>

      {/* Mobile Navigation */}
      <MobileNavbar />
    </div>
  );
};

export default AppLayout;
