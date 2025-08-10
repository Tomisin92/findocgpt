
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Route, BrowserRouter as Router, Routes } from 'react-router-dom';
import Header from './components/common/Header';
import Sidebar from './components/common/Sidebar';
import Dashboard from './pages/Dashboard';
import DocumentAnalysis from './pages/DocumentAnalysis';
import Forecasting from './pages/Forecasting';
import Home from './pages/Home';
import Strategy from './pages/Strategy';
import './styles/globals.css';

const queryClient = new QueryClient();

function App() {
    return (
        <QueryClientProvider client={queryClient}>
            <Router>
                <div className="min-h-screen bg-gray-50">
                    <Header />
                    <div className="flex">
                        <Sidebar />
                        <main className="flex-1 p-6">
                            <Routes>
                                <Route path="/" element={<Home />} />
                                <Route path="/documents" element={<DocumentAnalysis />} />
                                <Route path="/forecasting" element={<Forecasting />} />
                                <Route path="/strategy" element={<Strategy />} />
                                <Route path="/dashboard" element={<Dashboard />} />
                            </Routes>
                        </main>
                    </div>
                </div>
            </Router>
        </QueryClientProvider>
    );
}

export default App;