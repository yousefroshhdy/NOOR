
import { mockAuth, User } from './mockData';

// Storage keys
const TOKEN_KEY = 'drivepulse_auth_token';
const USER_KEY = 'drivepulse_user';

// Auth service functions
export const login = async (email: string, password: string): Promise<boolean> => {
  try {
    const { user, token } = await mockAuth.login(email, password);
    
    // Store auth data
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    
    return true;
  } catch (error) {
    console.error('Login failed:', error);
    return false;
  }
};

export const register = async (name: string, email: string, password: string): Promise<boolean> => {
  try {
    const { user, token } = await mockAuth.register(name, email, password);
    
    // Store auth data
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(user));
    
    return true;
  } catch (error) {
    console.error('Registration failed:', error);
    return false;
  }
};

export const logout = (): void => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
};

export const getCurrentUser = (): User | null => {
  const userJson = localStorage.getItem(USER_KEY);
  if (!userJson) return null;
  
  try {
    return JSON.parse(userJson) as User;
  } catch (error) {
    console.error('Failed to parse user data:', error);
    return null;
  }
};

export const isAuthenticated = (): boolean => {
  return !!localStorage.getItem(TOKEN_KEY) && !!getCurrentUser();
};

export const getAuthToken = (): string | null => {
  return localStorage.getItem(TOKEN_KEY);
};
