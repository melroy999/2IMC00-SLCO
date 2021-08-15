/**
 */
package poosl.tests;

import junit.framework.TestCase;

import junit.textui.TestRunner;

import poosl.PooslFactory;
import poosl.StatementList;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Statement List</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class StatementListTest extends TestCase {

	/**
	 * The fixture for this Statement List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected StatementList fixture = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(StatementListTest.class);
	}

	/**
	 * Constructs a new Statement List test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public StatementListTest(String name) {
		super(name);
	}

	/**
	 * Sets the fixture for this Statement List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void setFixture(StatementList fixture) {
		this.fixture = fixture;
	}

	/**
	 * Returns the fixture for this Statement List test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected StatementList getFixture() {
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
		setFixture(PooslFactory.eINSTANCE.createStatementList());
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

} //StatementListTest
