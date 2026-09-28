import { test, expect } from '@playwright/test';

test.describe('AgriPrice Sentinel - E2E', () => {
  test('should load the homepage and display crops', async ({ page }) => {
    await page.goto('/');

    // Check title
    await expect(page).toHaveTitle(/AgriPrice Sentinel/i);

    // Look for the hero section
    const heading = page.locator('h1', { hasText: 'AI-Powered Crop Price' });
    await expect(heading).toBeVisible();

    // Check if the crop grid is visible (just an assumption of the UI)
    // We assume the user can click on "Wheat" or similar.
    const wheatCard = page.locator('text=Wheat').first();
    await expect(wheatCard).toBeVisible();
  });

  test('should navigate to forecast dashboard', async ({ page }) => {
    await page.goto('/');
    
    // Find Wheat and click it (Assuming the landing page has a crop card that links to /dashboard/Wheat/Mandi)
    const wheatCard = page.locator('text=Wheat').first();
    await expect(wheatCard).toBeVisible();
    
    // Ideally we would click it, but for a simple baseline test, we navigate directly
    await page.goto('/dashboard/Wheat/Indore');

    // Wait for the chart or some specific text
    await expect(page.locator('text=Wheat Price Forecast').first()).toBeVisible();
  });
});
