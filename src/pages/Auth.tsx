
import { useState } from 'react';
import { Navigate } from 'react-router-dom';
import { isAuthenticated } from '@/services/authService';
import AuthLayout from '@/components/auth/AuthLayout';
import LoginForm from '@/components/auth/LoginForm';
import RegisterForm from '@/components/auth/RegisterForm';

const Auth = () => {
  const [isLogin, setIsLogin] = useState(true);

  // Redirect if already authenticated
  if (isAuthenticated()) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <AuthLayout 
      title="DrivePulse Tracker"
      subtitle={isLogin ? "Sign in to monitor your vehicles" : "Create an account to get started"}
    >
      {isLogin ? (
        <LoginForm onSwitchToRegister={() => setIsLogin(false)} />
      ) : (
        <RegisterForm onSwitchToLogin={() => setIsLogin(true)} />
      )}
    </AuthLayout>
  );
};

export default Auth;
