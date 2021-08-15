/**
 */
package slco.tests;

import junit.textui.TestRunner;

import slco.SlcoFactory;
import slco.UnidirectionalChannel;

/**
 * <!-- begin-user-doc -->
 * A test case for the model object '<em><b>Unidirectional Channel</b></em>'.
 * <!-- end-user-doc -->
 * @generated
 */
public class UnidirectionalChannelTest extends ChannelTest {

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static void main(String[] args) {
		TestRunner.run(UnidirectionalChannelTest.class);
	}

	/**
	 * Constructs a new Unidirectional Channel test case with the given name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public UnidirectionalChannelTest(String name) {
		super(name);
	}

	/**
	 * Returns the fixture for this Unidirectional Channel test case.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected UnidirectionalChannel getFixture() {
		return (UnidirectionalChannel)fixture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see junit.framework.TestCase#setUp()
	 * @generated
	 */
	@Override
	protected void setUp() throws Exception {
		setFixture(SlcoFactory.eINSTANCE.createUnidirectionalChannel());
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

} //UnidirectionalChannelTest
