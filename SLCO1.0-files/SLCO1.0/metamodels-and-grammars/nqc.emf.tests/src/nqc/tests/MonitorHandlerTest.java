/**
 */
package nqc.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import nqc.MonitorHandler;
import nqc.NqcFactory;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Monitor Handler</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class MonitorHandlerTest extends TestCase {

	/**
	 * The fixture for this Monitor Handler test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MonitorHandler fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(MonitorHandlerTest.class);
	}

	/**
	 * Constructs a new Monitor Handler test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MonitorHandlerTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Monitor Handler test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(MonitorHandler fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Monitor Handler test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MonitorHandler getFixture() {
		return fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(NqcFactory.eINSTANCE.createMonitorHandler());
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#tearDown()
	 * @generated
	 */
	@Override
	protected void tearDown() throws Exception {
		setFixture(null);
	}

} //MonitorHandlerTest
